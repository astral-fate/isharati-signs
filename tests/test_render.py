import imageio.v2 as imageio
import numpy as np
import pytest
from isharati.pose.render import SkeletonRenderer

pytestmark = pytest.mark.slow


def test_render_writes_playable_mp4_with_one_frame_per_pose(tmp_path, lexicon):
    pose = lexicon.keypoints(lexicon.lookup("وجه"))[:30]
    pose = np.concatenate([pose, pose])[:30]
    out = SkeletonRenderer(size=256).render(pose, tmp_path / "a.mp4")
    assert out.exists() and out.stat().st_size > 0
    r = imageio.get_reader(out)
    assert r.count_frames() == 30
    assert r.get_data(0).shape == (256, 256, 3)
