# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import array_ops
import tensorflow as tf


class Bidirectional(rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""
    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNCells.
            Args:
              cells: list of RNNCells that will be composed in this order.
              state_is_tuple: If True, accepted and returned states are n-tuples, where
                `n = len(cells)`.  If False, the states are all
                concatenated along the column axis.  This latter behavior will soon be
                deprecated.
            Raises:
              ValueError: if cells is empty (not allowed)
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    # # From Tensorflow
    # def __call__(self, inputs, state, scope=None):
    #     """Run this multi-layer cell on inputs, starting from state."""
    #     with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
    #         cur_state_pos = 0
    #         cur_inp = inputs  # Shape: (batch_size, embedding)
    #         new_states = []
    #         print("###### States to be run on ######")
    #         print(state)
    #         print("---------------------------------")
    #         for i, cell in enumerate(self._cells):
    #             print(i, cell)
    #             print(inputs)
    #             print("#####--------------#####")
    #             print("#####--------------#####")
    #             with vs.variable_scope("Cell%d" % i):
    #                 if not nest.is_sequence(state):  # Checks if state is NOT a tuple
    #                     raise ValueError("Expected state to be a tuple of length %d, but received: %s"
    #                                      % (len(self.state_size), state))
    #                 cur_state = state[i]
    #                 cur_inp, new_state = cell(cur_inp, cur_state)
    #                 new_states.append(new_state)
    #     new_states = tuple(new_states)
    #     return cur_inp, new_states


    # Silinho and Siminho tried this
    def __call__(self, inputs, state, scope=None):
        """Run this bidirecional cell on inputs, starting from state."""
        with vs.variable_scope(scope or type(self).__name__):
            cur_inp = inputs  # Shape: (batch_size, embedding)

            new_states = []

            # Expecting that there are only two cells. Need to either check (or create them here in stead)

            # Forward processing
            fwd_m_out_list = []
            fwd_cell = self._cells[0]
            with vs.variable_scope("Cell%d" % 0):
                if not nest.is_sequence(state):  # Checks if state is NOT a tuple
                    raise ValueError("Expected state to be a tuple of length %d, but received: %s"
                                     % (len(self.state_size), state))
                cur_state = state[0]
                fwd_m_out, fwd_state_out = fwd_cell(cur_inp, cur_state)
                fwd_m_out_list.append(fwd_m_out)

                new_states.append(fwd_state_out)

            # Backward processing
            bwd_m_out_list = []
            bwd_cell = self._cells[1]
            cur_reversed_inputs = array_ops.reverse(fwd_m_out, [True, False])  # Output_dim = 1, input_dim = 0
            with vs.variable_scope("Cell%d" % 1):
                if not nest.is_sequence(state):  # Checks if state is NOT a tuple
                    raise ValueError("Expected state to be a tuple of length %d, but received: %s"
                                     % (len(self.state_size), state))
                cur_state = state[1]
                bwd_m_out, bwd_state_out = bwd_cell(cur_reversed_inputs, cur_state)
                bwd_m_out_list.append(bwd_m_out)

                new_states.append(bwd_state_out)

        new_states = (tuple(new_states) if self._state_is_tuple else array_ops.concat(1, new_states))
        # print(fwd_m_out_list)
        # flat_output_fw = nest.flatten(fwd_m_out_list)
        # flat_output_bw = nest.flatten(bwd_m_out_list)
        # print("1")
        # print(flat_output_bw)
        #
        # flat_outputs = tuple(
        #     array_ops.concat([fw, bw], 1)
        #     for fw, bw in zip(flat_output_fw, flat_output_bw))
        # print("2")


        # nest.pack_sequence_as:
        # `flat_sequence` converted to have the same recursive structure as `structure`
        outputs_fwd = nest.pack_sequence_as(structure=fwd_m_out,
                                        flat_sequence=fwd_m_out_list)
        outputs_bwd = nest.pack_sequence_as(structure=bwd_m_out,
                                            flat_sequence=bwd_m_out_list)

        m_output = array_ops.concat(1, outputs_fwd + outputs_bwd)
        print(m_output)
        #m_output = bwd_m_out
        return m_output, new_states


