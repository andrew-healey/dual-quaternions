from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import Self,Tuple
import math

eps=1e-5

@dataclass
class Quaternion:
    data:ndarray

    @staticmethod
    def from_vector(vector:ndarray)->Self:
        return Quaternion(data=np.concatenate([
            np.array([0]),
            vector,
        ]))
    
    @staticmethod
    def from_axis_angle(axis:ndarray,angle:float)->Self:
        a = math.cos(angle/2)
        v = math.sin(angle/2) * axis/np.linalg.norm(axis)
        return Quaternion(data=np.concatenate([
            np.array([a]),
            v
        ]))

    @property
    def a(self):
        return self.data[0]
    @property
    def v(self):
        return self.data[1:]

    def __add__(self,other):
        return Quaternion(
            data=self.data+other.data
        )

    def __mul__(self,other):

        if(type(other)==float or type(other) == int): return Quaternion(data=self.data*other)

        # import pdb;pdb.set_trace()
        assert type(other)==Quaternion,f"type(other)=={type(other)},other=={other}"
        a = self.a*other.a - self.v @ other.v
        v = self.a*other.v + other.a*self.v - np.cross(self.v,other.v)
        return Quaternion(
            data=np.concatenate([np.array([a]),v])
        )
    
    def __rmul__(self,other):
        if(type(other)==float or type(other) == int): return self*other
        raise TypeError(f"Tried to right-multiply Quaternion by {type(other)}")
    
    def __div__(self,other):
        return self * 1/other
    
    def conj(self):
        return Quaternion(
            data=np.concatenate([self.a[None],-self.v])
        )
    
    def norm(self):
        return np.linalg.norm(self.data)
    
    def normalize(self):
        return self * 1/self.norm()

@dataclass
class DualQuaternion:
    qr: Quaternion
    qd: Quaternion

    @staticmethod
    def from_vector(vector:ndarray)->Self:
        r = Quaternion(data=np.array([1,0,0,0]))
        return DualQuaternion.from_rot_trans(
            rotation=r,
            translation=vector*2
        )

    @staticmethod
    def from_rot_trans(rotation:Quaternion,translation:ndarray)->Self:
        t = Quaternion(data=np.concatenate([np.array([0]),translation]))
        print("t",t)
        return DualQuaternion(
            qr=rotation,
            qd=0.5 * t * rotation
        )
    
    def transform_vector(self,vector:ndarray)->ndarray:
        v = DualQuaternion.from_vector(vector)
        print("v",v)
        print("q",self)
        print(self*v)
        print(self*v*self.conj_negative())
        v_p = self * v * self.conj_negative()
        qid = Quaternion.from_axis_angle(np.array([1,0,0]),0)
        dqid = DualQuaternion.from_rot_trans(qid,np.zeros(3))
        for i in range(5):
            v_p = dqid * v_p * dqid.conj_negative()
        print("v_p",v_p)
        qr,trans = v_p.to_rot_trans()
        return trans/2
    
    def to_rot_trans(self)->Tuple[Quaternion,ndarray]:
        assert abs(self.norm()-1)<eps,f"Qr is not normalized! ||Qr|| = {self.qr.norm()}"

        translation = 2 * (self.qd * self.qr.conj()).v
        return self.qr,translation

    def __add__(self,other):
        return DualQuaternion(
            qr=self.qr+other.qr,
            qd=self.qd+other.qd,
        )

    def __mul__(self,other):
        if(type(other)==float or type(other) == int): return DualQuaternion(qr=self.qr*other,qd=self.qd*other)

        assert type(other)==DualQuaternion,f"type(other)=={type(other)}"
        return DualQuaternion(
            qr=self.qr * other.qr,
            qd=self.qr * other.qd + self.qd * other.qr
        )

    def __rmul__(self,other):
        if(type(other)==float or type(other) == int): return self*other
        raise TypeError(f"Tried to right-multiply DualQuaternion by {type(other)}")
    
    def conj(self):
        return DualQuaternion(
            qr=self.qr.conj(),
            qd=self.qd.conj()
        )
    
    def conj_negative(self):
        return DualQuaternion(
            qr=self.qr.conj(),
            qd=-1*self.qd.conj()
        )
    
    def norm(self):
        return self.qr.norm()
    
    def normalize(self):
        return self*1/self.norm()