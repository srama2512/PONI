#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""PolicyRegistry is extended from habitat.Registry to provide
registration for policies used for transfer.

Import the policy registry object using

.. code:: py

    from policies.policy_registry import policy_registry

Various decorators for registry different kind of classes with unique keys

-   Register a policy: ``@policy_registry.register_policy``
"""

from typing import Optional

from habitat.core.registry import Registry


class PolicyRegistry(Registry):
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL policy with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        """
        return cls._register_impl("policy", to_register, name)

    @classmethod
    def get_policy(cls, name: str):
        r"""Get the RL policy with :p:`name`."""
        return cls._get_impl("policy", name)


policy_registry = PolicyRegistry()
