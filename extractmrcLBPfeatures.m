function [features] = extractmrcLBPfeatures(RGB)

R=RGB(:,:,1);
G=RGB(:,:,2);
B=RGB(:,:,3);
R=imadjust(R);
G=imadjust(G);
B=imadjust(B);

R1=extractLBPFeatures(R,'Upright',false,'Radius',1);
G1=extractLBPFeatures(G,'Upright',false,'Radius',1);
B1=extractLBPFeatures(B,'Upright',false,'Radius',1);
R3=extractLBPFeatures(R,'Upright',false,'Radius',3);
G3=extractLBPFeatures(G,'Upright',false,'Radius',3);
B3=extractLBPFeatures(B,'Upright',false,'Radius',3);
R9=extractLBPFeatures(R,'Upright',false,'Radius',9);
G9=extractLBPFeatures(G,'Upright',false,'Radius',9);
B9=extractLBPFeatures(B,'Upright',false,'Radius',9);
R27=extractLBPFeatures(R,'Upright',false,'Radius',27);
G27=extractLBPFeatures(G,'Upright',false,'Radius',27);
B27=extractLBPFeatures(B,'Upright',false,'Radius',27);

features=[R1 R3 R9 R27 G1 G3 G9 G27 B1 B3 B9 B27];
%features=[R1 R3 R9 G1 G3 G9 B1 B3 B9];
%features=[G1 G3 G9 G27];
end

