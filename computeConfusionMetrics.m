function [precision,recall,specificity,accuracy,F1,MCC] = computeConfusionMetrics(confusionMatrixNormalizedValues)
%COMPUTECONFUSIONMETRICS Summary of this function goes here
%   Detailed explanation goes here
TP=confusionMatrixNormalizedValues(4);
FP=confusionMatrixNormalizedValues(3);
FN=confusionMatrixNormalizedValues(2);
TN=confusionMatrixNormalizedValues(1);
precision=(TP/(TP+FP));
recall=(TP/(TP+FN));
specificity=(TN/(TN+FP));
accuracy=((TP+TN)/(TP+FP+FN+TN));
F1=2*(precision*recall)/(precision+recall);
MCC=((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
end

