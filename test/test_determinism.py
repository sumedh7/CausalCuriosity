import pybullet
import pytest
import numpy as np

from cem_planning.envs.block_manipulation_env import BlockManipulationEnv, SCENARIOS
import pybullet_utils.bullet_client as bc


class TestDeterminism:

    @pytest.mark.parametrize('scenario', SCENARIOS)
    def test_determinism(self, scenario):
        # Load environment, run action sequence, reset, run same action sequence.
        # Check that rewards and other things such as states are exactly the same
        env = BlockManipulationEnv(
            bullet_client=bc.BulletClient(connection_mode=pybullet.DIRECT),
                                          scenario=scenario)

        env.reset()

        action_sequence = [env.action_space.sample() for _ in range(30)]
        save_state = env.bullet_client.saveState
        restore_state = env.bullet_client.restoreState

        initial_state = save_state()
        rewards_1 = []
        finger_positions_1 = []
        for action in action_sequence:
            _, r, _, _ = env.step(action)
            rewards_1.append(r)
            finger_positions_1.append(env.finger._get_latest_observation().position)


        restore_state(initial_state)
        rewards_2 = []
        finger_positions_2 = []
        for action in action_sequence:
            _, r, _, _ = env.step(action)
            rewards_2.append(r)
            finger_positions_2.append(env.finger._get_latest_observation().position)

        np.testing.assert_almost_equal(rewards_1, rewards_2)
        np.testing.assert_almost_equal(finger_positions_1, finger_positions_2)
        # assert  rewards_1 == rewards_2
