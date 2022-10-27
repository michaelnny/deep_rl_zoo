# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The functions has been modified by The Deep RL Zoo Authors
# to support PyTorch operation.
#
# ============================================================================
"""Tests for value_learning."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
import numpy as np
from deep_rl_zoo import value_learning


class QLearningTest(absltest.TestCase):
    """Abstract base class for Q learning RL value ops tests."""

    def setUp(self):
        super(QLearningTest, self).setUp()
        self.q_tm1 = torch.tensor([[1, 1, 0], [1, 2, 0]], dtype=torch.float32)
        self.q_t = torch.tensor([[0, 1, 0], [1, 2, 0]], dtype=torch.float32)
        self.a_tm1 = torch.tensor([0, 1], dtype=torch.int64)
        self.discount_t = torch.tensor([0, 1], dtype=torch.float32)
        self.r_t = torch.tensor([1, 1], dtype=torch.float32)
        self.qlearning = value_learning.qlearning(self.q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t)

    def testRankCheck(self):
        q_tm1 = torch.tensor([0.0])
        with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
            self.qlearning = value_learning.qlearning(q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t)

    def testCompatibilityCheck(self):
        a_tm1 = torch.tensor([3])
        with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
            self.qlearning = value_learning.qlearning(self.q_tm1, a_tm1, self.r_t, self.discount_t, self.q_t)

    def testTarget(self):
        np.testing.assert_allclose(self.qlearning.extra.target.numpy(), np.array([1, 3]))

    def testTDError(self):
        np.testing.assert_allclose(self.qlearning.extra.td_error.numpy(), np.array([0, 1]))

    def testLoss(self):
        # Loss is 0.5 * td_error^2
        np.testing.assert_allclose(self.qlearning.loss.numpy(), np.array([0, 0.5]))

    def _manually_calculate_loss_with_autograd(self, q_tm1):
        # Manually calculate loss, as torch does not have function like tf.gradients.
        qa_tm1 = q_tm1.gather(-1, self.a_tm1[..., None]).squeeze(-1)
        target_q_t = self.qlearning.extra.target
        loss = 0.5 * torch.square(target_q_t - qa_tm1)

        # Take gradients of the negative loss, so that the tests here check the
        # values propogated during gradient _descent_, rather than _ascent_.

        # Set grad_outputs to avoid output gradients become a scalar.
        grad_q_tm1 = torch.autograd.grad(outputs=-loss, inputs=q_tm1, grad_outputs=torch.ones_like(loss))

        return grad_q_tm1

    def testGradQtm1(self):
        q_tm1 = torch.autograd.Variable(self.q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(q_tm1)
        np.testing.assert_allclose(grad_q_tm1[0].numpy(), np.array([[0, 0, 0], [0, 1, 0]]))

    def testNoOtherGradients(self):
        # Gradients are only defined for q_tm1, not any other input.
        # Bellman residual variants could potentially generate a gradient wrt q_t.
        q_tm1 = torch.autograd.Variable(self.q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(q_tm1)

        no_grads = [self.q_t.grad, self.r_t.grad, self.a_tm1.grad, self.discount_t.grad]
        self.assertEqual(no_grads, [None for _ in no_grads])


class DoubleQLearningTest(absltest.TestCase):
    """Abstract base class for DoubleQ learning RL value ops tests."""

    def setUp(self):
        super(DoubleQLearningTest, self).setUp()
        self.q_tm1 = torch.tensor([[1, 1, 0], [1, 2, 0]], dtype=torch.float32)
        self.a_tm1 = torch.tensor([0, 1], dtype=torch.int64)
        self.discount_t = torch.tensor([0, 1], dtype=torch.float32)
        self.r_t = torch.tensor([1, 1], dtype=torch.float32)
        # The test is written so that it calculates the same thing as QLearningTest:
        # The selector, despite having different values, select the same actions,
        self.q_t_selector = torch.tensor([[2, 10, 1], [11, 20, 1]], dtype=torch.float32)
        # whose values are unchanged. (Other values are changed and larger.)
        self.q_t_value = torch.tensor([[99, 1, 98], [91, 2, 66]], dtype=torch.float32)
        self.double_qlearning = value_learning.double_qlearning(
            self.q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t_value, self.q_t_selector
        )

    def testRankCheck(self):
        q_t_selector = torch.tensor([0.0])
        with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
            self.double_qlearning = value_learning.double_qlearning(
                self.q_tm1, self.a_tm1, self.r_t, self.discount_t, self.q_t_value, q_t_selector
            )

    def testCompatibilityCheck(self):
        r_t = torch.tensor([3])
        with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
            self.double_qlearning = value_learning.double_qlearning(
                self.q_tm1, self.a_tm1, r_t, self.discount_t, self.q_t_value, self.q_t_selector
            )

    def testDoubleQLearningBestAction(self):
        np.testing.assert_allclose(self.double_qlearning.extra.best_action.numpy(), np.array([1, 1]))

    def testDoubleQLearningTarget(self):
        np.testing.assert_allclose(self.double_qlearning.extra.target.numpy(), np.array([1, 3]))

    def testDoubleQLearningTDError(self):
        np.testing.assert_allclose(self.double_qlearning.extra.td_error.numpy(), np.array([0, 1]))

    def testDoubleQLearningLoss(self):
        # Loss is 0.5 * td_error^2
        np.testing.assert_allclose(self.double_qlearning.loss.numpy(), np.array([0, 0.5]))

    def _manually_calculate_loss_with_autograd(self, q_tm1):
        # Manually calculate loss, as torch does not have function like tf.gradients.
        qa_tm1 = q_tm1.gather(-1, self.a_tm1[..., None]).squeeze(-1)
        target_q_t = self.double_qlearning.extra.target
        loss = 0.5 * torch.square(target_q_t - qa_tm1)

        # Take gradients of the negative loss, so that the tests here check the
        # values propogated during gradient _descent_, rather than _ascent_.

        # Set grad_outputs to avoid output gradients become a scalar.
        grad_q_tm1 = torch.autograd.grad(outputs=-loss, inputs=q_tm1, grad_outputs=torch.ones_like(loss))

        return grad_q_tm1

    def testGradQtm1(self):
        q_tm1 = torch.autograd.Variable(self.q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(q_tm1)
        np.testing.assert_allclose(grad_q_tm1[0].numpy(), np.array([[0, 0, 0], [0, 1, 0]]))

    def testDoubleQLearningNoOtherGradients(self):
        # Gradients are only defined for q_tm1, not any other input.
        # Bellman residual variants could potentially generate a gradient wrt q_t.

        q_tm1 = torch.autograd.Variable(self.q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(q_tm1)

        no_grads = [self.r_t.grad, self.a_tm1.grad, self.discount_t.grad, self.q_t_value.grad, self.q_t_selector.grad]

        self.assertEqual(no_grads, [None for _ in no_grads])


class CategoricalDistRLTest(absltest.TestCase):
    """Abstract base class for Distributional RL value ops tests."""

    def setUp(self):
        super(CategoricalDistRLTest, self).setUp()
        # Define both state- and action-value transitions here for the different

        # learning rules tested in the subclasses.

        self.atoms_tm1 = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        self.atoms_t = torch.clone(self.atoms_tm1)

        self.logits_q_tm1 = torch.tensor(
            [
                [[1, 1, 1], [0, 9, 9], [0, 9, 0], [0, 0, 0]],
                [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
                [[1, 1, 1], [0, 9, 9], [0, 0, 0], [0, 9, 0]],
                [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
                [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
            ],
            dtype=torch.float32,
        )
        self.logits_q_t = torch.tensor(
            [
                [[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]],
            ],
            dtype=torch.float32,
        )
        # mean Q_t are approximately:
        #  1.0 1.0 0.5 1.5
        #  0.75 0.5 1.0 0.5
        #  1.0 1.0 1.5 0.5
        #  0.75 0.5 1.0 0.5
        #  0.75 0.5 1.25 0.5

        self.logits_v_tm1 = torch.tensor([[0, 9, 0], [9, 0, 9], [0, 9, 0], [9, 9, 0], [9, 0, 9]], dtype=torch.float32)
        self.logits_v_t = torch.tensor([[0, 0, 9], [1, 1, 1], [0, 0, 9], [1, 1, 1], [0, 9, 9]], dtype=torch.float32)

        self.a_tm1 = torch.tensor([2, 1, 3, 0, 1], dtype=torch.int64)
        self.r_t = torch.tensor([0.5, 0.0, 0.5, 0.8, -0.1], dtype=torch.float32)
        self.discount_t = torch.tensor([0.8, 1.0, 0.8, 0.0, 1.0], dtype=torch.float32)

    def assertEachInputRankAndCompatibilityChecked(self, nt, inputs, invalid_inputs):
        """Check class constructor raises exception if an input tensor is invalid.

        Args:
        nt: namedtuple to be tested.
        inputs: list of (valid) inputs to class constructor.
        invalid_inputs: list of invalid alternative inputs. Should be of same
            length as `inputs`, so that each input can be swapped out for a broken
            input individually.
        """
        for i, alt_input in enumerate(invalid_inputs):
            broken_inputs = list(inputs)
            broken_inputs[i] = alt_input
            with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
                nt(*broken_inputs)

    def _manually_calculate_loss_with_autograd(self, logits_q_tm1, target_q_t):
        # Manually calculate loss, as torch does not have function like tf.gradients.
        # logit_qa_tm1 = logits_q_tm1.gather(1, self.a_tm1[None, ..., None]).squeeze(-1)
        logit_qa_tm1 = logits_q_tm1[torch.arange(0, logits_q_tm1.shape[0]), self.a_tm1]

        loss = F.cross_entropy(input=logit_qa_tm1, target=target_q_t, reduction='none')

        # Take gradients of the negative loss, so that the tests here check the
        # values propogated during gradient _descent_, rather than _ascent_.

        # Set grad_outputs to avoid output gradients become a scalar.
        grad_q_tm1 = torch.autograd.grad(outputs=-loss, inputs=logits_q_tm1, grad_outputs=torch.ones_like(loss))

        return grad_q_tm1


class CategoricalDistQLearningTest(CategoricalDistRLTest):
    def setUp(self):
        super(CategoricalDistQLearningTest, self).setUp()

        self.inputs = [
            self.atoms_tm1,
            self.logits_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.atoms_t,
            self.logits_q_t,
        ]
        self.qlearning = value_learning.categorical_dist_qlearning(*self.inputs)

    def testRankCheck(self):
        alt_inputs = [torch.tensor(False, dtype=torch.bool) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(value_learning.categorical_dist_qlearning, self.inputs, alt_inputs)

    def testCompatibilityCheck(self):
        alt_inputs = [torch.tensor([1]) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(value_learning.categorical_dist_qlearning, self.inputs, alt_inputs)

    def testTarget(self):
        # Target is projected KL between r_t + pcont_t atoms_t and
        # probabilities corresponding to logits_q_tm1 [ a_tm1 ].
        expected = np.array([[0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 1.0], [0.4, 0.6, 0.0], [0.1, 0.5, 0.4]])
        np.testing.assert_allclose(self.qlearning.extra.target.numpy(), expected, atol=1e-3)

    def testLoss(self):
        # Loss is CE between logits_q_tm1 [a_tm1] and target.
        expected = np.array([9.0, 3.69, 9.0, 0.69, 5.19])
        np.testing.assert_allclose(self.qlearning.loss.numpy(), expected, atol=1e-2)

    def testGradQtm1(self):
        logits_q_tm1 = torch.autograd.Variable(self.logits_q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(logits_q_tm1, self.qlearning.extra.target)
        grad_q_tm1 = grad_q_tm1[0].numpy()
        expected = np.zeros_like(grad_q_tm1)
        expected[0, 2] = [-1, -1, 1]
        expected[1, 1] = [-1, 1, -1]
        expected[2, 3] = [-1, -1, 1]
        expected[3, 0] = [-1, 1, -1]
        expected[4, 1] = [-1, 1, -1]

        np.testing.assert_allclose(np.sign(grad_q_tm1), expected)

    def testNoOtherGradients(self):
        # Gradients are only defined for q_tm1, not any other input.
        # Bellman residual variants could potentially generate a gradient wrt q_t.
        logits_q_tm1 = torch.autograd.Variable(self.logits_q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(logits_q_tm1, self.qlearning.extra.target)

        no_grads = [
            self.logits_q_t.grad,
            self.r_t.grad,
            self.a_tm1.grad,
            self.discount_t.grad,
            self.atoms_t.grad,
            self.atoms_tm1.grad,
        ]
        self.assertEqual(no_grads, [None for _ in no_grads])


class CategoricalDistDoubleQLearningTest(CategoricalDistRLTest):
    def setUp(self):
        super(CategoricalDistDoubleQLearningTest, self).setUp()

        self.q_t_selector = torch.tensor(
            [[0, 2, 0, 5], [0, 1, 2, 1], [0, 2, 5, 0], [0, 1, 2, 1], [1, 2, 3, 1]], dtype=torch.float32
        )

        self.inputs = [
            self.atoms_tm1,
            self.logits_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.atoms_t,
            self.logits_q_t,
            self.q_t_selector,
        ]
        self.qlearning = value_learning.categorical_dist_double_qlearning(*self.inputs)

    def testRankCheck(self):
        alt_inputs = [torch.tensor(False, dtype=torch.bool) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(
            value_learning.categorical_dist_double_qlearning, self.inputs, alt_inputs
        )

    def testCompatibilityCheck(self):
        alt_inputs = [torch.tensor([1]) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(
            value_learning.categorical_dist_double_qlearning, self.inputs, alt_inputs
        )

    def testTarget(self):
        # Target is projected KL between r_t + pcont_t atoms_t and
        # probabilities corresponding to logits_q_tm1 [ a_tm1 ].
        expected = np.array([[0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 1.0], [0.4, 0.6, 0.0], [0.1, 0.5, 0.4]])
        np.testing.assert_allclose(self.qlearning.extra.target.numpy(), expected, atol=1e-3)

    def testLoss(self):
        # Loss is CE between logits_q_tm1 [a_tm1] and target.
        expected = np.array([9.0, 3.69, 9.0, 0.69, 5.19])
        np.testing.assert_allclose(self.qlearning.loss.numpy(), expected, atol=1e-2)

    def testGradQtm1(self):
        logits_q_tm1 = torch.autograd.Variable(self.logits_q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(logits_q_tm1, self.qlearning.extra.target)
        grad_q_tm1 = grad_q_tm1[0].numpy()
        expected = np.zeros_like(grad_q_tm1)
        expected[0, 2] = [-1, -1, 1]
        expected[1, 1] = [-1, 1, -1]
        expected[2, 3] = [-1, -1, 1]
        expected[3, 0] = [-1, 1, -1]
        expected[4, 1] = [-1, 1, -1]

        np.testing.assert_allclose(np.sign(grad_q_tm1), expected)

    def testNoOtherGradients(self):
        # Gradients are only defined for q_tm1, not any other input.
        # Bellman residual variants could potentially generate a gradient wrt q_t.
        logits_q_tm1 = torch.autograd.Variable(self.logits_q_tm1, requires_grad=True)
        grad_q_tm1 = self._manually_calculate_loss_with_autograd(logits_q_tm1, self.qlearning.extra.target)

        no_grads = [
            self.logits_q_t.grad,
            self.r_t.grad,
            self.a_tm1.grad,
            self.discount_t.grad,
            self.atoms_t.grad,
            self.atoms_tm1.grad,
            self.q_t_selector.grad,
        ]
        self.assertEqual(no_grads, [None for _ in no_grads])


class QuantileRegressionLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.dist_src = torch.tensor([[-1.0, 3.0], [-1.0, 3.0]])
        self.tau_src = torch.tensor([[0.2, 0.7], [0.0, 0.4]])
        self.dist_target = torch.tensor([[-3.0, 4.0, 2.0], [-3.0, 4.0, 2.0]])

        # delta = [[ -2  5  3 ], [ -6  1 -1 ]]
        # Huber(2.)-delta = [[  2  8  4 ], [ 10 .5 .5 ]]
        #
        # First batch element:
        # |tau - Id_{d<0}| = [[ .8 .2 .2 ], [ .3 .7 .3 ]]
        # Loss = 1/3 sum( |delta| . |tau - Id_{d<0}| )  = 2.0
        # Huber(2.)-loss = 2.5
        #
        # Second batch element:
        # |tau - Id_{d<0}| = [[ 1. 0. 0. ], [ .6 .4 .6 ]]
        # Loss = 2.2
        # Huber(2.)-loss = 8.5 / 3
        self.expected_loss = {0.0: torch.tensor([2.0, 2.2]), 2.0: torch.tensor([2.5, 8.5 / 3.0])}

    @parameterized.named_parameters(('nohuber', 0.0), ('huber', 2.0))
    def test_quantile_regression_loss_batch(self, huber_param):
        """Tests for a full batch."""
        # Compute quantile regression loss.
        actual = value_learning._quantile_regression_loss(
            self.dist_src, self.tau_src, self.dist_target, huber_param=huber_param
        )
        # Test outputs in batch.
        np.testing.assert_allclose(actual, self.expected_loss[huber_param], rtol=3e-7)


class QuantileLearningTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

        # n_batch = 3, n_taus = 2, n_actions = 4
        self.dist_q_tm1 = torch.tensor(
            [[[0, 1, -5, 6], [-1, 3, 0, -2]], [[-5, 9, -5, 6], [2, 3, 1, -4]], [[5, 1, -5, 7], [-1, 3, 0, -2]]],
            dtype=torch.float32,
        )

        self.tau_q_tm1 = torch.tensor([[0.2, 0.7], [0.1, 0.5], [0.3, 0.4]], dtype=torch.float32)

        self.a_tm1 = torch.tensor([1, 2, 0], dtype=torch.int64)
        self.r_t = torch.tensor([0.5, -1.0, 0.0], dtype=torch.float32)
        self.discount_t = torch.tensor([0.5, 0.0, 1], dtype=torch.float32)

        self.dist_q_t = torch.tensor(
            [[[0, 5, 2, 2], [0, -3, 2, 2]], [[-3, -1, 4, -3], [1, 3, 1, -4]], [[-2, 2, -5, -7], [1, 3, 2, -2]]],
            dtype=torch.float32,
        )

        self.q_t_selector = torch.tensor(
            [[[0, 7, 2, -2], [0, 4, 2, 2]], [[-3, -1, 4, 3], [1, 3, 1, 4]], [[-1, -2, -5, -6], [-1, -5, 2, -2]]],
            dtype=torch.float32,
        )

        # Scenario 1: no double Q, bootstrap target selected over mean dist_q_t
        dist_qa_tm1 = torch.tensor([[1, 3], [-5, 1], [5, -1]], dtype=torch.float32)
        # dist_qa_tm1                                      [ 1,  3]
        #     (batch x n_tau)                          =   [-5,  1]
        #                                                  [ 5, -1]
        # dist_q_t[mean]                                   [ 0.0,   1.0,   2.0,   2.0]
        #     (batch x n_actions)                      =   [-1.0,   1.0,   2.5,  -3.5]
        #                                                  [-0.5,   2.5,  -1.5,  -4.5]
        # a_t = argmax_a dist_q_t[mean]                    [2]
        #     (batch)                                  =   [2]
        #                                                  [1]
        # dist_qa_t                                        [2, 2]
        #     (batch x n_taus)                         =   [4, 1]
        #                                                  [2, 3]
        # target = r + gamma * dist_qa_t                   [ 1.5, 1.5]
        #     (batch x n_taus)                         =   [-1,  -1]
        #                                                  [ 2,   3]
        dist_target = torch.tensor([[1.5, 1.5], [-1, -1], [2, 3]], dtype=torch.float32)

        # Use qr loss to compute expected results (itself tested explicitly above class QuantileRegressionLossTest).
        self.expected = {}
        for huber_param in [0.0, 1.0]:
            self.expected[huber_param] = value_learning._quantile_regression_loss(
                dist_qa_tm1,
                self.tau_q_tm1,
                dist_target,
                huber_param,
            )

        # Scenario 2: using double Q, these match the q_t_selector above

        dist_qa_tm1 = torch.tensor([[1, 3], [-5, 1], [5, -1]], dtype=torch.float32)
        # dist_qa_tm1                                      [ 1,  3]
        #     (batch x n_tau)                          =   [-5,  1]
        #                                                  [ 5, -1]
        # q_t_selector[mean]                               [ 0.0,  5.5,  2.0,  0.0]
        #     (batch x n_actions)                      =   [-1.0,  1.0,  2.5,  3.5]
        #                                                  [-1.0, -3.5, -1.5, -4.0]
        # a_t = argmax_a q_t_selector                      [1]
        #     (batch)                                  =   [3]
        #                                                  [0]
        # dist_qa_t                                        [ 5, -3]
        #     (batch x n_taus)                         =   [-3, -4]
        #                                                  [-2,  1]
        # target = r + gamma * dist_qa_t                   [ 3, -1]
        #     (batch x n_taus)                         =   [-1, -1]
        #                                                  [-2,  1]
        double_q_dist_target = torch.tensor([[3, -1], [-1, -1], [-2, 1]], dtype=torch.float32)

        # Use qr loss to compute expected results (itself tested explicitly above class QuantileRegressionLossTest).
        self.double_q_expected = {}
        for huber_param in [0.0, 1.0]:
            self.double_q_expected[huber_param] = value_learning._quantile_regression_loss(
                dist_qa_tm1,
                self.tau_q_tm1,
                double_q_dist_target,
                huber_param,
            )

        self.inputs = [
            self.dist_q_tm1,
            self.tau_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.dist_q_t,
        ]

        self.double_q_inputs = [
            self.dist_q_tm1,
            self.tau_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.dist_q_t,
            self.q_t_selector,
        ]

    def assertEachInputRankAndCompatibilityChecked(self, nt, inputs, invalid_inputs):
        """Check class constructor raises exception if an input tensor is invalid.

        Args:
        nt: namedtuple to be tested.
        inputs: list of (valid) inputs to class constructor.
        invalid_inputs: list of invalid alternative inputs. Should be of same
            length as `inputs`, so that each input can be swapped out for a broken
            input individually.
        """
        for i, alt_input in enumerate(invalid_inputs):
            broken_inputs = list(inputs)
            broken_inputs[i] = alt_input
            with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
                nt(*broken_inputs)

    def testRankCheck(self):
        alt_inputs = [torch.tensor(False, dtype=torch.bool) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(value_learning.quantile_q_learning, self.inputs, alt_inputs)

    def testCompatibilityCheck(self):
        alt_inputs = [torch.tensor([1]) for _ in self.inputs]
        self.assertEachInputRankAndCompatibilityChecked(value_learning.quantile_q_learning, self.inputs, alt_inputs)

    def testRankCheckDoublQ(self):
        alt_inputs = [torch.tensor(False, dtype=torch.bool) for _ in self.double_q_inputs]
        self.assertEachInputRankAndCompatibilityChecked(
            value_learning.quantile_double_q_learning, self.double_q_inputs, alt_inputs
        )

    def testCompatibilityCheckDoublQ(self):
        alt_inputs = [torch.tensor([1]) for _ in self.double_q_inputs]
        self.assertEachInputRankAndCompatibilityChecked(
            value_learning.quantile_double_q_learning, self.double_q_inputs, alt_inputs
        )

    @parameterized.named_parameters(('nohuber', 0.0), ('huber', 1.0))
    def test_quantile_q_learning_batch(self, huber_param):
        """Tests for a full batch."""
        # Test outputs.
        actual = value_learning.quantile_q_learning(
            self.dist_q_tm1,
            self.tau_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.dist_q_t,
            huber_param,
        ).loss
        np.testing.assert_allclose(self.expected[huber_param].numpy(), actual.numpy(), rtol=1e-5)

    @parameterized.named_parameters(('nohuber', 0.0), ('huber', 1.0))
    def test_quantile_double_q_learning_batch(self, huber_param):
        """Tests for a full batch."""
        # Test outputs.
        actual = value_learning.quantile_double_q_learning(
            self.dist_q_tm1,
            self.tau_q_tm1,
            self.a_tm1,
            self.r_t,
            self.discount_t,
            self.dist_q_t,
            self.q_t_selector,
            huber_param,
        ).loss
        np.testing.assert_allclose(self.double_q_expected[huber_param].numpy(), actual.numpy(), rtol=1e-5)


class RetraceTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self._lambda = 0.9

        # Swap axies to make time major, since most of the agents use [T, B] convension.
        self._qs = np.array(
            [[[1.1, 2.1], [-1.1, 1.1], [3.1, -3.1], [-1.2, 0.0]], [[2.1, 3.1], [9.5, 0.1], [-2.1, -1.1], [0.1, 7.4]]],
            dtype=np.float32,
        ).swapaxes(0, 1)
        self._targnet_qs = np.array(
            [[[1.2, 2.2], [-1.2, 0.2], [2.2, -1.2], [-2.25, -6.0]], [[4.2, 2.2], [1.2, 1.2], [-1.2, -2.2], [1.5, 1.0]]],
            dtype=np.float32,
        ).swapaxes(0, 1)
        self._actions = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.int32).swapaxes(0, 1)
        self._rewards = np.array([[-1.3, -1.3, 2.3, 42.0], [1.3, 5.3, -3.3, -5.0]], dtype=np.float32).swapaxes(0, 1)
        self._pcontinues = np.array([[0.0, 0.89, 0.85, 0.99], [0.88, 1.0, 0.83, 0.95]], dtype=np.float32).swapaxes(0, 1)
        self._target_policy_probs = np.array(
            [[[0.5, 0.5], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]], [[0.1, 0.9], [1.0, 0.0], [0.3, 0.7], [0.7, 0.3]]],
            dtype=np.float32,
        ).swapaxes(0, 1)
        self._behavior_policy_probs = np.array([[0.5, 0.1, 0.9, 0.3], [0.4, 0.6, 1.0, 0.9]], dtype=np.float32).swapaxes(0, 1)

        self.expected = np.array(
            [[2.8800001, 3.8934109, 4.5942383], [3.1121615e-1, 2.0253206e1, 3.1601219e-3]], dtype=np.float32
        ).swapaxes(0, 1)

    def test_retrace_batch(self):
        """Tests for a full batch."""
        # Test outputs.
        actual = value_learning.retrace(
            torch.tensor(self._qs[:-1, ...], dtype=torch.float32),
            torch.tensor(self._targnet_qs[1:, ...], dtype=torch.float32),
            torch.tensor(self._actions[:-1, ...], dtype=torch.long),
            torch.tensor(self._actions[1:, ...], dtype=torch.long),
            torch.tensor(self._rewards[:-1, ...], dtype=torch.float32),
            torch.tensor(self._pcontinues[:-1, ...], dtype=torch.float32),
            torch.tensor(self._target_policy_probs[1:, ...], dtype=torch.float32),
            torch.tensor(self._behavior_policy_probs[1:, ...], dtype=torch.float32),
            self._lambda,
        )
        np.testing.assert_allclose(self.expected, actual.loss.numpy(), rtol=1e-5)


if __name__ == '__main__':
    absltest.main()
