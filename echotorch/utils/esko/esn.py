# -*- coding: utf-8 -*-
#
# File : utils/helpers/ESN.py
# Description : Helper class for ESN classifier
# Date : 27th of April, 2021
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Imports


# ESN base class
class ESN:
    """ESN base class

    ESN base class

    Parameters
    ----------

    Attributes
    ----------
    """

    # region CONSTRUCTORS

    def __init__(
            self
    ):
        pass
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # History
    @property
    def history(self):
        """History (getter)
        TODO
        """
        return None
    # end history

    # History (setter)
    @history.setter
    def history(self, value):
        """History (setter)
        TODO
        """
        pass
    # end history

    # Default callbacks (getter)
    @property
    def _default_callbacks(self):
        """Default callbacks (getter)
        TODO
        """
        return None
    # end _default_callbacks

    # endregion PROPERTIES

    # region PUBLIC

    # Get default callbacks
    def get_default_callbacks(self):
        """Get default callbacks
        TODO
        """
        return None
    # end get_default_callbacks

    # Notify
    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in ``method_name`` with
        parameters specified in `cb_kwargs``.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_batch_begin
        * on_batch_end

        """
        # Call the method
        getattr(self, method_name)(self, **cb_kwargs)

        # Call each callback
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)
        # end for
    # end notify

    # On train begin
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        """On train begin
        TODO
        """
        pass
    # end on_train_begin

    # On train end
    def on_train_end(self, net, X=None, y=None, **kwargs):
        """On train end
        TODO
        """
        pass
    # end on_train_end

    # On batch begin
    def on_batch_begin(self, net, Xi=None, yi=None, training=False, **kwargs):
        """On batch begin
        TODO
        """
        pass
    # end on_batch_begin

    # On batch end
    def on_batch_end(self, net, Xi=None, yi=None, training=False, **kwargs):
        """On batch end
        TODO
        """
        pass
    # end on_batch_end

    # Initialize callbacks
    def initialize_callbacks(self):
        """Initializes all callbacks and save the results in the
        ``callbacks_`` attribute.
        TODO
        """
        pass
    # end initialize_callbacks

    # Initialize module
    def initialize_module(self):
        """Initialize the module.

        Note that if the module has learned parameters, those will be
        reset.
        TODO
        """
        pass
    # end initialize_module

    # endregion PUBLIC

    # region PRIVATE

    # Yield callbacks
    def _yield_callbacks(self):
        """Yield all callbacks
        TODO
        """
        pass
    # end _yield_callbacks

    # endregion PRIVATE

# end ESN
