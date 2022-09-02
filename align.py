
from skimage import feature, measure, transform, exposure
import numpy


class Aligner:

    # based on
    # https://stackoverflow.com/questions/62280342/image-alignment-with-orb-and-ransac-in-scikit-image#62332179

    def __init__(self, reference):

        ref = exposure.equalize_adapthist(reference)
        self.reference = ref
        self.de = feature.SIFT()

        self.de.detect_and_extract(ref)

        self.ref_keypoints = self.de.keypoints
        self.ref_descriptors = self.de.descriptors

    def align(self, target):
        "Align the target image to the reference image"

        brightened = exposure.equalize_adapthist(target)
        self.de.detect_and_extract(brightened)
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
            max_trials=1000
            )
        rot = transform.EuclideanTransform(
            rotation=transform_robust.rotation
            )
        trans = transform.EuclideanTransform(
            translation=-numpy.flip(transform_robust.translation)
            )
        robust = rot + trans
        warped = transform.warp(target,
                                robust.inverse,
                                order=1,
                                mode="constant",
                                cval=0,
                                clip=True,
                                preserve_range=True
                                )
        return warped.astype(target.dtype)
