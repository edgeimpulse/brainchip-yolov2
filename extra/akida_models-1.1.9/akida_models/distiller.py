#!/usr/bin/env python
# *****************************************************************************
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Tools for Knowledge Distillation training.

Originated from https://keras.io/examples/vision/knowledge_distillation/.

Reference Hinton et al. (2015) https://arxiv.org/abs/1503.02531
"""

import tensorflow as tf
from functools import partial
from tensorflow import GradientTape
from keras import Model
from keras.losses import KLDivergence, CategoricalCrossentropy


class Distiller(Model):
    """ The class that will be used to train the student model using the
    distillation knowledge method.

    Reference `Hinton et al. (2015) <https://arxiv.org/abs/1503.02531>`_.

    Args:
        student (keras.Model): the student model
        teacher (keras.Model): the well trained teacher model
        alpha (float, optional): weight to student_loss_fn and 1-alpha
            to distillation_loss_fn. Defaults to 0.1
    """

    def __init__(self, student, teacher, alpha=0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.alpha = alpha

    @property
    def base_model(self):
        return self.student

    @property
    def layers(self):
        return self.base_model.layers

    def compile(self,
                optimizer,
                metrics,
                student_loss_fn,
                distillation_loss_fn):
        """ Configure the distiller.

        Args:
            optimizer (keras.optimizers.Optimizer): Keras optimizer
                for the student weights
            metrics (keras.metrics.Metric): Keras metrics for evaluation
            student_loss_fn (keras.losses.Loss): loss function of difference
                between student predictions and ground-truth
            distillation_loss_fn (keras.losses.Loss): loss function of
                difference between student predictions and teacher predictions
        """

        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        with GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions, student_predictions)
            loss = self.alpha * student_loss + (1 -
                                                self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        })
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def save(self, *args, **kwargs):
        return self.base_model.save(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        return self.base_model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.base_model.load_weights(*args, **kwargs)


class DeitDistiller(Distiller):
    """Distiller class to train the student model using the
    Knowledge Distillation (KD) method, found on https://arxiv.org/pdf/2012.12877.pdf

    The main difference with the classic KD is that the student has to produce two potential
    classification outputs. This type of training is based on the assumption that each output
    has sufficiently interacted with the whole model, therefore the main architecture can be
    trained through two different sources, as follows:

        >>> output, output_kd = student(input)
        >>> output_tc = teacher(input)
        >>> student_loss = student_loss_fn(y_true, output)
        >>> distillation_loss = distillation_loss_fn(output_tc, output_kd)

    This means we will expect to have different inputs for each loss, unlike classical KD,
    where the student's prediction is shared for both losses. However, given that each
    classifier has interacted with the student model, the gradient of each loss will contribute
    to the update of the model weights according to the alpha percentage.

    Args:
        student (keras.Model): the student model
        teacher (keras.Model): the well trained teacher model
        alpha (float, optional): weight to student_loss_fn and 1-alpha
            to distillation_loss_fn. Defaults to 0.1
        temperature (float, optional): if ``distiller_type`` when compile is equal to 'soft',
            this value will be used as temperature parameter of KLDistillationLoss.
            Defaults to 1.0.
    """

    def __init__(self, student, *args, temperature=1.0, **kwargs):
        assert len(student.outputs) == 2, "Student must be a multi-output model, with 2 outputs"

        # Append an output with the sum of heads
        y = tf.math.add_n(student.outputs) / 2
        _student = Model(student.inputs, student.outputs + [y], name=student.name)

        super().__init__(_student, *args, **kwargs)
        self._student = student
        self.temperature = temperature

    @property
    def base_model(self):
        return self._student

    def compile(self, optimizer, metrics, student_loss_fn, distiller_type):
        """ Configure the distiller.

        Args:
            optimizer (keras.optimizers.Optimizer): Keras optimizer
                for the student weights
            metrics (keras.metrics.Metric): Keras metrics for evaluation
            student_loss_fn (keras.losses.Loss): loss function of difference
                between student predictions and ground-truth
            distiller_type (str): loss function type to define the difference
                between student predictions and teacher-truth, within ['soft', 'hard', 'none'] which
                will result in performing KLDistillationLoss, CategoricalCrossentropy or
                student_loss_fn only respectively.
        """
        assert distiller_type in ['soft', 'hard', 'none']

        def _loss_forward(y_true, y_pred, loss_fn, index=0, **kwargs):
            if isinstance(y_pred, (tuple, list)):
                y_pred = y_pred[index]
            return loss_fn(y_true, y_pred, **kwargs)

        def _compile_distillation_loss_fn():
            if distiller_type == "soft":
                distillation_loss_fn = KLDistillationLoss(temperature=self.temperature)
            else:
                # Follow https://arxiv.org/pdf/2012.12877.pdf, this variant takes the
                # hard decision of the teacher as a true label. Therefore, we add the
                # prediction encoder, as well as a label smoothing equal to 0.1
                y = tf.math.softmax(self.teacher.outputs[0], axis=-1)
                self.teacher = Model(self.teacher.inputs, y, name=self.teacher.name)
                distillation_loss_fn = CategoricalCrossentropy(
                    from_logits=True, label_smoothing=0.1)
            return partial(_loss_forward, loss_fn=distillation_loss_fn, index=1)

        if distiller_type == "none" or self.teacher is None:
            # In this case, we just train the first output of student
            self.teacher = distillation_loss_fn = None
            self.student = Model(self.student.inputs,
                                 self.student.outputs[0], name=self.student.name)
            self.student.compile(optimizer, student_loss_fn, metrics)
        else:
            distillation_loss_fn = _compile_distillation_loss_fn()
            student_loss_fn = partial(_loss_forward, loss_fn=student_loss_fn)
            super().compile(optimizer, metrics, student_loss_fn, distillation_loss_fn)

    def _update_metrics(self, metrics):
        # Rename keys in the result dictionary for a more explicit display
        return {k.replace('output_1_', 'head_').replace('output_2_', 'dist_head_')
                 .replace('output_3_', ''): v for k, v in metrics.items()}

    def train_step(self, data):
        if self.teacher is None:
            return self.student.train_step(data)
        return self._update_metrics(super().train_step(data))

    def test_step(self, data):
        if self.teacher is None:
            return self.student.test_step(data)
        return self._update_metrics(super().test_step(data))


class KLDistillationLoss(KLDivergence):
    """
    The `KLDistillationLoss` is a simple wrapper around the KLDivergence loss
    that accepts raw predictions instead of probability distributions.

    Before invoking the KLDivergence loss, it converts the inputs predictions to
    probabilities by dividing them by a constant 'temperature' and applies a
    softmax.

    Args:
        temperature (float): temperature for softening probability
            distributions. Larger temperature gives softer distributions.
    """

    def __init__(self, temperature=3):
        self.temperature = temperature
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Following https://github.com/facebookresearch/deit/blob/main/losses.py#L63
        # The result of KLDivergence must be scaled
        scale_factor = tf.constant(self.temperature ** 2, dtype=tf.float32)
        return super().__call__(
            tf.nn.softmax(y_true / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1)) * scale_factor
