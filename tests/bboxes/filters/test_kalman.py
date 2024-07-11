# # Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# #
# # MIT License
# from __future__ import annotations

# from cv2ext.bboxes.filters import KalmanFilter
# import hypothesis.strategies as st
# from hypothesis import given


# def test_kalman_filter_no_change():
#     bbox1 = (50, 50, 100, 100)
#     kfilter = KalmanFilter(bbox1)
#     for _ in range(100):
#         bbox2 = kfilter(bbox1)
#         assert bbox2 == bbox1

# @given(
#     delta=st.integers(min_value=1, max_value=100),
#     iters=st.integers(min_value=1, max_value=100),
# )
# def test_kalman_filter_constant_change(delta: int, iters: int):
#     bbox1 = (50, 50, 100, 100)
#     kfilter = KalmanFilter(bbox1)
#     bbox2 = (0, 0, 0, 0)
#     for _ in range(iters):
#         bbox2 = kfilter(bbox1)
#         for (b1, b2) in zip(bbox1, bbox2):
#             assert b2 <= b1
#         bbox1 = (bbox1[0] + delta, bbox1[1] + delta, bbox1[2] + delta, bbox1[3] + delta)
#     for (b1, b2) in zip(bbox1, bbox2):
#         assert abs(b1 - b2) <= iters * delta
