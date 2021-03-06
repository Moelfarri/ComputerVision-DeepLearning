import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    x1_i = max(prediction_box[0],gt_box[0])
    y1_i = max(prediction_box[1],gt_box[1])
    x2_i = min(prediction_box[2],gt_box[2])
    y2_i = min(prediction_box[3],gt_box[3])
    
    if x1_i > x2_i or y1_i > y2_i:
        intersection = 0
    else:
        intersection = (x2_i-x1_i)*(y2_i-y1_i)
        

    # Compute union
    x1, y1, x2, y2 = prediction_box
    x1g,y1g,x2g,y2g = gt_box 
    box_area1 = np.abs((x2-x1)*(y2-y1))
    box_area2 = np.abs((x2g-x1g)*(y2g-y1g))
    union     = box_area1 + box_area2 - intersection
    
    
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    
    return num_tp/(num_tp + num_fp)



def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    
    return num_tp/(num_tp+num_fn)
    


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    best_matches = []  # iou, match (for sorting later)
    for prediction_box in prediction_boxes:
        max_iou = -1
        best_match = max_iou, prediction_box, None
        for gt_box in gt_boxes:
            iou = calculate_iou(prediction_box, gt_box)
            if iou > max_iou and iou >= iou_threshold:
                max_iou = iou
                best_match = max_iou, prediction_box, gt_box
        best_matches.append(best_match)

    # Sort all matches on IoU in descending order
    best_matches.sort(key=lambda x: x[0])
    p_boxes = [e[1] for e in best_matches]
    g_boxes = [e[2] for e in best_matches]
    return np.array(p_boxes), np.array(g_boxes)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    p_boxes_matched, g_boxes_matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
 
    confusion_stats = dict()
    confusion_stats['true_pos']  = 0
    confusion_stats['false_pos'] = 0
    confusion_stats['false_neg'] = 0

    # count ground-truth match found for prediction box
    #False positive - prediction makes a false prediction
    for p_box, g_box in zip(p_boxes_matched, g_boxes_matched):
        if g_box is None:
            confusion_stats['false_pos'] += 1
        else:
            confusion_stats['true_pos'] += 1

    #False negative - prediction does not predict a gt_box that exists
    for gt_box in gt_boxes:
        gt_box_covered = False
        for i in range(len(g_boxes_matched)):
            gt_box_covered = all(gt_box == g_boxes_matched[i])
            if gt_box_covered:
                break #when it break it doesnt read last line
        confusion_stats['false_neg'] += not gt_box_covered



    assert confusion_stats['true_pos'] + confusion_stats['false_pos'] == len(prediction_boxes)
 
    return confusion_stats

 


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0
    fn = 0
    
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        confusion_stats = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        tp += confusion_stats['true_pos']
        fp += confusion_stats['false_pos']
        fn += confusion_stats['false_neg']
        
    p = calculate_precision(tp, fp, fn)
    r = calculate_recall(tp, fp, fn)
    return float(p), float(r)



def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    for confidence_threshold in confidence_thresholds:
        # idea: keep only prediction boxes if score >= confidence_threshold
        # loops thru images (has to be done as the dimensions vary - # boxes per image!
        confident_pbs = []
        for box_scores, predicted_boxes in zip(confidence_scores, all_prediction_boxes):
            index = box_scores >= confidence_threshold
            pb = predicted_boxes[index]
            confident_pbs.append(pb)
        precision, recall = calculate_precision_recall_all_images(confident_pbs, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    ys = []
    recall_levels = np.linspace(1.0, 0, 11)
    for recall_level in recall_levels:  # from right to left (increasing steps)
        index = recalls >= recall_level  # no upper limit (steal max values from the right)
        if any(index):
            ys.append(max(precisions[index]))
        else:
            ys.append(0)
    return sum(ys) / len(ys)


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
