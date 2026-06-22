from rlinf.envs.realworld.common.camera.zed_camera import ZEDCamera


class _FakeSL:
    class ERROR_CODE:
        SUCCESS = "SUCCESS"
    class SVO_COMPRESSION_MODE:
        H264 = "H264"
    class RecordingParameters:
        def __init__(self, path, compression):
            self.path = path
            self.compression = compression


def _bare_zed():
    z = ZEDCamera.__new__(ZEDCamera)
    z._sl = _FakeSL
    calls = {}

    class FakeCam:
        def enable_recording(self, rp):
            calls["enable"] = rp
            return _FakeSL.ERROR_CODE.SUCCESS
        def disable_recording(self):
            calls["disable"] = True

    z._camera = FakeCam()
    z._recording = False
    z._calls = calls
    return z


def test_start_recording_enables_with_h264_path():
    z = _bare_zed()
    z.start_recording("/tmp/ep0_wrist_1.svo2")
    assert z._calls["enable"].path == "/tmp/ep0_wrist_1.svo2"
    assert z._calls["enable"].compression == _FakeSL.SVO_COMPRESSION_MODE.H264
    assert z._recording is True


def test_stop_recording_disables_and_is_idempotent():
    z = _bare_zed()
    z.stop_recording()  # not recording -> safe no-op
    assert "disable" not in z._calls
    z.start_recording("/tmp/x.svo2")
    z.stop_recording()
    assert z._calls["disable"] is True
    assert z._recording is False


def test_open_with_retry_recovers_after_transient_failure():
    calls = {"open": 0, "close": 0}

    class FakeCam:
        def open(self, ip):
            calls["open"] += 1
            return "FAIL" if calls["open"] == 1 else _FakeSL.ERROR_CODE.SUCCESS
        def close(self):
            calls["close"] += 1

    status = ZEDCamera._open_with_retry(
        FakeCam(), None, _FakeSL, "36443134", attempts=3, settle_s=0.0
    )
    assert status == _FakeSL.ERROR_CODE.SUCCESS
    assert calls["open"] == 2  # failed once, succeeded on retry
    assert calls["close"] == 1  # closed before the retry


def test_open_with_retry_gives_up_after_attempts():
    class FakeCam:
        def open(self, ip):
            return "FAIL"
        def close(self):
            pass

    status = ZEDCamera._open_with_retry(
        FakeCam(), None, _FakeSL, "36443134", attempts=3, settle_s=0.0
    )
    assert status == "FAIL"  # returns last status; caller raises
