# Phase 4: defect-type classifier

## What was done

A rejected part now gets a defect type. `vqi.defect_type` crops the original image around
the peak of the anomaly map and classifies the crop with an exported OpenVINO model, using
only the plain runtime, as the anomaly model does. `scripts/train_classifier.py` builds the
crops for every leather defect, compares a fine-tuned ResNet18 with a HOG + SVM baseline by
5-fold stratified cross-validation, writes the pooled results, confusion matrices and
per-image predictions to `results/classifier/leather/`, and exports a ResNet18 trained on
all images to `exports/classifier/leather/`.

The API loads the classifier named in the serve config and runs it on NOK parts only. The
answer carries the defect type and its probability; the decision log stores the type in
the column that had been waiting for it since Phase 6. `scripts/daily_report.py` writes a
second CSV with each day's NOK parts split by type.

## Why

The crop comes from the anomaly map, not from a ground-truth mask, because the line has no
masks. Training on mask crops and serving map crops would test a classifier on inputs it
never saw. Measured first: the map peak falls outside the mask for some color and cut
defects, but a 256 px crop around it reached the defect in every leather image, so the
crop is that wide.

Leather has 92 defects, about 18 per class. A single held-out split would leave a handful
of parts per class and a result that depends on the split, so both models are judged by
cross-validation, every image predicted by a model that never saw it, with the class count
next to every recall. ResNet18's settings were fixed before the first run because there is
no data left to tune them on without leaking; the SVM's single setting is chosen inside
each training fold. HOG + SVM is the classical baseline: it shows how much a pretrained CNN
adds, and, working on grey crops, how much the colour class depends on colour.

ResNet18 got 89 of 92 right and HOG + SVM 59, so the CNN is served. The result is honest
only within its size: one error moves a class by about five points, and the classifier
knows only leather's five classes, with no "unknown" answer. Its probability is returned
but not calibrated.

## Bugs found on the way

OpenCV 5, which the project pins, has no `HOGDescriptor` any more; the import failed at
the first test run. HOG now comes from scikit-image, already installed with anomalib, and
is imported only inside the feature function, so the API image does not need it (D55).

A mutation check showed that the summary test could not tell a correct count from the
largest count in a row, because in its example every class was mostly right. The example
now has a class with no correct prediction.

## Summary

The pipeline is now hybrid: the anomaly model, trained on good parts only, decides and
locates; a small supervised classifier names what it found, from the same crop the line
would see. On leather the classifier is clearly better than the classical baseline, but 92
images are too few to call its accuracy precise, and a new product needs its own labelled
defects. Phase 7 adds the operator panel and the final README.
