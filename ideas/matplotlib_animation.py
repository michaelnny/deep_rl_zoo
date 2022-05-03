# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
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
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
(plot,) = plt.plot([], [])


def init_function():
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 250)
    return (plot,)


def Redraw_Function(UpdatedVal):
    new_x = np.arange(500) * UpdatedVal
    new_y = np.arange(500) ** 2 * UpdatedVal
    plot.set_data(new_x, new_y)
    return (plot,)


# Animated_Figure = FuncAnimation(fig, Redraw_Function, init_func=init_function, frames=np.arange(1, 5, 1), interval=1000)
Animated_Figure = FuncAnimation(fig, Redraw_Function, init_func=init_function, frames=np.arange(1, 20, 1), interval=1000)
plt.show()
