% Script for extracting textural features from the processed images.


clear mrcLBPfeatures
clc
Xtrainidx=[];
Xtestidx=[];
Xvalididx=[];
selpath=uigetdir;
png_files=dir(fullfile(selpath,'*.png'));
bowman_png_files=dir(fullfile(selpath,'*_capsule*.png'));
[C,png_files_indices]=setdiff({png_files.name},{bowman_png_files.name});
png_files=png_files(png_files_indices);
for png_num=1:size(png_files)
    [filepath,filename,ext]=fileparts(png_files(png_num).name);
    RGB=imread(strcat(selpath,'/',png_files(png_num).name));
    grayscale=rgb2gray(RGB);
    adjgrayscale=imadjust(grayscale);
    features=extractmrcLBPfeatures(RGB);
    mrcLBPfeatures(png_num,:)=features;
    
    %features1=extractLBPFeatures(rgb2gray(RGB),'Upright',false,'Radius',1);
    %features3=extractLBPFeatures(rgb2gray(RGB),'Upright',false,'Radius',3);
    %features9=extractLBPFeatures(rgb2gray(RGB),'Upright',false,'Radius',9);
    %features27=extractLBPFeatures(rgb2gray(RGB),'Upright',false,'Radius',27);
    %HOGfeatures=extractHOGFeatures(rgb2gray(RGB));
    %LBPfeatures(png_num,:)=[features1 features3 features9 features27];
    
    %CMYK=rgb2cmyk(RGB);
    %M=CMYK(:,:,2);
    %features1=extractLBPFeatures(M,'Upright',false,'Radius',1);
    %features3=extractLBPFeatures(M,'Upright',false,'Radius',3);
    %features9=extractLBPFeatures(M,'Upright',false,'Radius',9);
    %features27=extractLBPFeatures(M,'Upright',false,'Radius',27);
    %MagentaLBPfeatures(png_num,:)=[features1 features3 features9 features27];
 
    
    glcm1=graycomatrix(adjgrayscale, 'offset', [0 1], 'Symmetric', true);
    glcm2=graycomatrix(adjgrayscale, 'offset', [-1 1], 'Symmetric', true);
    glcm3=graycomatrix(adjgrayscale, 'offset', [-1 0], 'Symmetric', true);
    glcm4=graycomatrix(adjgrayscale, 'offset', [-1 -1], 'Symmetric', true);
    haralick1=haralickTextureFeatures(glcm1);
    %haralickfeatures=haralick1;
    haralick2=haralickTextureFeatures(glcm2);
    haralick3=haralickTextureFeatures(glcm3);
    haralick4=haralickTextureFeatures(glcm4);
    haralick1234=[haralick1 haralick2 haralick3 haralick4];
    medieharalick=mean(haralick1234,2);
    rangeharalick=range(haralick1234,2);
    haralickfeatures=[medieharalick; rangeharalick];
    allHaralickFeatures(png_num,:)=haralickfeatures';
end

%X=mrcLBPfeatures;
%Xtraining=X([Xtrainidx; Xvalididx],:);
%Xtest=X(Xtestidx,:);
%Xtrainingscaled=zscore(Xtraining);