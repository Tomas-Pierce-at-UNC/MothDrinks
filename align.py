
from skimage import feature, measure, transform, exposure, util
import numpy


class Aligner:

    # based on
    # https://stackoverflow.com/questions/62280342/image-alignment-with-orb-and-ransac-in-scikit-image#62332179

    __slots__ = ("reference", "de", "ref_keypoints", "ref_descriptors")

    def __init__(self, reference):

        self.reference = exposure.equalize_adapthist(reference)
        self.de = feature.SIFT()

        self.de.detect_and_extract(self.reference)

        self.ref_keypoints = self.de.keypoints
        self.ref_descriptors = self.de.descriptors

    def align(self, target):
        "Align the target image to the reference image"

        btarget = exposure.equalize_adapthist(target)
        self.de.detect_and_extract(btarget)
        keypoints = self.de.keypoints
        descriptors = self.de.descriptors
        matches = feature.match_descriptors(
            self.ref_descriptors,
            descriptors,
            cross_check=True)
        ref_matches = self.ref_keypoints[matches[:, 0]]
        matches = keypoints[matches[:, 1]]
        transform_robust, inliers = measure.ransac(
            (ref_matches, matches),
            transform.EuclideanTransform,
            min_samples=5,
            residual_threshold=0.5,
            max_trials=500
            )
        robust = transform.EuclideanTransform(
            rotation=transform_robust.rotation,
            translation=-numpy.flip(transform_robust.translation)
            )

        warped = transform.warp(target,
                                robust.inverse,
                                order=1,
                                mode="constant",
                                cval=0,
                                clip=True,
                                preserve_range=True
                                )
        return warped.astype(target.dtype)


def test_main():
    ex1 = "data//moth26_2022-02-15_freeflight.cine"
    from cine import Cine
    import tube2
    from skimage import io as skio
    from matplotlib import pyplot
    import time
    cin = Cine(ex1)
    t_i = time.time()
    med = cin.get_video_median()
    t_f = time.time()
    print("median time", t_f - t_i)
    h = cin.get_ith_image(100)
    cin.close()
    left, right = tube2.get_tube(med)
    res_med = med[:, left:right]
    t1 = time.time()
    ally = Aligner(res_med)
    t0 = time.time()
    aligned = ally.align(h)
    t1 = time.time()
    print(t1 - t0)
    # skio.imshow(aligned)
    # pyplot.show()
    return aligned


if __name__ == "__main__":
    al = test_main()
