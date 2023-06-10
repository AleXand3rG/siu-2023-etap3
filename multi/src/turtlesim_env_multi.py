# encoding: utf8
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

from turtlesim_env_base import TurtlesimEnvBase


class TurtlesimEnvSingle(TurtlesimEnvBase):
    def __init__(self):
        super().__init__()

    def setup(self, routes_fname: str, agent_cnt=None):
        super().setup(routes_fname, agent_cnt)
        for agent in self.agents.values():  # liczba kroków - indywidualnie dla każdego agenta
            agent.step_sum = 0

    def reset(self, tnames=None, sections='default'):
        ret = super().reset(tnames, sections)
        if tnames is None:
            tnames = self.agents.keys()
        for tname in tnames:
            self.agents[tname].step_sum = 0  # liczba kroków zerowana wybiórczo
        return ret

    # TODO-STUDENCI przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2
    def students_step(self, action, tname):
        twist = Twist()
        twist.linear.x = action[0]
        twist.linear.y = 0
        twist.angular.z = action[1]

        self.set_twist_velocity(tname, twist)
        self.wait_after_move()
        self.set_twist_velocity(tname, twist)
        self.wait_after_move()

    def set_twist_velocity(self, tname, twist):
        self.tapi.setVel(tname, twist)

    def wait_after_move(self):
        rospy.sleep(self.WAIT_AFTER_MOVE)

    def step(self, actions, realtime=False):  # {id_żółwia:(prędkość,skręt)}

        # pozycja PRZED krokiem sterowania
        for tname, action in actions.items():
            self.step_sum += 1
            agent = self.agents[tname]
            agent.pose = self.tapi.getPose(tname)
            _, _, _, agent.fd, _, _ = self.get_road(tname)  # odległość do celu (mogła ulec zmianie)

        # action: [prędkość,skręt]
        if realtime:  # jazda+skręt+jazda+skręt
            self.students_step(action=action, tname=tname)

        else:  # skok+obrót
            for tname, action in actions.items():
                # obliczenie i wykonanie przesunięcia
                pose = self.agents[tname].pose
                vx = np.cos(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
                vy = np.sin(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
                p = Pose(x=pose.x + vx, y=pose.y + vy, theta=pose.theta + action[1])
                self.tapi.setPose(tname, p, mode='absolute')
            rospy.sleep(self.WAIT_AFTER_MOVE)

        collisions = self.tapi.getColisions(self.agents.keys(), self.COLLISION_DIST)
        colliding = set()  # nazwy kolidujących agentów
        for collision in collisions:
            colliding.add(collision['name1'])
            colliding.add(collision['name2'])

        # pozycje i sytuacje PO kroku sterowania
        result_dict = {}  # {tname:(get_map(),reward,done)}
        for tname in actions:
            done = False  # flaga wykrytego końca scenariusza symulacji

            pose = self.agents[tname].pose  # położenie przed ruchem
            pose1 = self.tapi.getPose(tname)  # położenie po ruchu
            self.agents[tname].pose = pose1

            fx1, fy1, fa1, fd1, _, _ = self.get_road(tname)  # warunki drogowe po przemieszczeniu
            vx1 = (pose1.x - pose.x) / self.SEC_PER_STEP  # aktualna prędkość - składowa x
            vy1 = (pose1.y - pose.y) / self.SEC_PER_STEP  # -"-                   y
            v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)  # aktualny moduł prędkości
            fv1 = np.sqrt(fx1 ** 2 + fy1 ** 2)  # zalecany moduł prędkości

            # wyznaczenie składników funkcji celu
            r1 = min(0, self.SPEED_FINE_RATE * (v1 - fv1))  # kara za przekroczenie prędkości
            r2 = 0
            if fv1 > .001:
                vf1 = (vx1 * fx1 + vy1 * fy1) / fv1  # rzut prędkości faktycznej na zalecaną
                if vf1 > 0:
                    r2 = self.SPEED_RWRD_RATE * vf1  # nagroda za jazdę z prądem
                else:
                    r2 = -self.SPEED_RVRS_RATE * vf1  # kara za jazdę pod prąd
            r3 = self.DIST_RWRD_RATE * (self.agents[tname].fd - fd1)  # nagroda za zbliżenie się do celu
            r4 = 0

            if abs(fx1) + abs(fy1) < .01 and fa1 == 1:  # wylądowaliśmy poza trasą
                r4 = self.OUT_OF_TRACK_FINE
                done = True

            # Detekcja kolizji
            r5 = 0
            if self.DETECT_COLLISION \
                    and self.get_map(tname)[6][self.GRID_RES // 2, self.GRID_RES - 1] == 0 \
                    and tname in colliding:
                r5 = self.OUT_OF_TRACK_FINE
                done = True

            reward = fa1 * (r1 + r2) + r3 + r4 + r5
            # sp=speed, fl=flow, cl=closing, tr=track, col=collision
            # print(f'RWD: {reward:.2f} = {fa1:.2f}*(sp{r1:.2f} fl{r2:.2f}) cl{r3:.2f} tr{r4:.2f} col{r5:.2f}')
            if self.agents[tname].step_sum > self.MAX_STEPS:
                done = True

            result_dict[tname] = (self.get_map(tname), reward, done)

        return result_dict


def provide_env():
    """Przygotowanie środowiska dla symulacji jednoagentowej"""
    return TurtlesimEnvSingle()
