
from copy import deepcopy
from typing import Iterable
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion

class FrankaEasyIK():
    def __init__(self):
        self.robot = rtb.models.Panda()
        self.last_q = None

    def __call__(self, p: Iterable[float], q: Iterable[float] = [1., 0., 0., 0.], verbose=False) -> Iterable[float]:
        """ do custom inverse kinematics

        Args:
            p (Float[3]): Cartesian Position
            q (Float[4]): Absolute Quaternion Orienation w.r.t. robot base
                - quternion notation: w,x,y,z
            verboser (bool): Print results
            
        Raises:
            Exception: When IK not found

        Returns:
            Float[7]: 7 DoF robot joint configuration
        """
        assert len(p) == 3, f"position length: {len(p)} != 3"
        assert len(q) == 4, f"quaternion length: {len(q)} != 4"
        
        p = deepcopy(p)
        q = deepcopy(q)
        
        sol = self.robot.ikine_LM(SE3.Trans(*p) * UnitQuaternion(np.array(q)).SE3(),start='panda_link0', end="panda_hand",q0=self.last_q,ilimit=5000,slimit=100,tol=1e-10)
        q1 = sol.q
        succ = sol.success
        reason = sol.reason
        
        if not succ:
            raise Exception(f"IK not found because: {reason}")
        if verbose:
            print("last q before: ", self.last_q)
        self.last_q = q1
        if verbose:
            print("last q: ", self.last_q)
        return np.array(q1)
if __name__ == "__main__":
    ik = FrankaEasyIK()

    position = [0.5,0.,0.3] # x, y, z
    orientation = [1., 0., 0., 0.] # x, y, z, w
    q = ik(position, orientation)
    print(q)