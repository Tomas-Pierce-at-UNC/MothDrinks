#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:25:48 2023

@author: tomas
"""

from skimage import feature, measure, transform, exposure, util
import numpy

class SiftAligner:
    
    # based on
    # https://stackoverflow.com/questions/62280342/image-alignment-with-orb-and-ransac-in-scikit-image#62332179


    def __init__(self, reference):
        "Creates an aligner that will align images to a reference image"
        self.reference = exposure.equalize_adapthist(reference)
        self.reference = reference
        self.de = feature.SIFT(n_octaves=4)

        self.de.detect_and_extract(self.reference)

        self.ref_keypoints = self.de.keypoints
        self.ref_descriptors = self.de.descriptors
        
    
    def find_transform(self, target):
        """Find a transformation that aligns target to the reference.
        Returns None if one is not found"""

        btarget = exposure.equalize_adapthist(target)
        #btarget = target
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
            max_trials=250
            )
        if transform_robust is None:
            return None
        robust = transform.EuclideanTransform(
            rotation=transform_robust.rotation,
            translation=-numpy.flip(transform_robust.translation)
            )
        return robust
    
    def apply_transform(self, target, transform):
        "Applies the given transform to the target"
        warped = transform.warp(target,
                                transform.inverse,
                                order=1,
                                mode="constant",
                                cval=0,
                                clip=True,
                                preserve_range=True
                                )
        return warped.astype(target.dtype)