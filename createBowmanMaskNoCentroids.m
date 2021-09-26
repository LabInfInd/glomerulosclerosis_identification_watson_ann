% Script for creating the masks of the bowman capsule.

clear
clc
selpath=uigetdir;
png_files=dir(fullfile(selpath,'*.png'));
bowman_png_files=dir(fullfile(selpath,'*_capsule*.png'));
[C,png_files_indices]=setdiff({png_files.name},{bowman_png_files.name});
png_files=png_files(png_files_indices);
for png_num=1:size(png_files)
    [filepath,filename,ext]=fileparts(png_files(png_num).name);
    RGB=imread(strcat(selpath,'/',png_files(png_num).name));
    grayimg=rgb2gray(RGB);
    CMYK=rgb2cmyk(RGB);
    Lab=rgb2lab(RGB);
    G=RGB(:,:,2);
    M=CMYK(:,:,2);
    ab=Lab(:,:,2:3);
    
    %Green
    adj=imadjust(G);
    bin=adj>190;
    %threshold=graythresh(adj);
    %bin=imbinarize(adj,threshold);
    er=imerode(bin,strel('disk',1));
    dil=imdilate(er,strel('disk',1));
    filt=medfilt2(dil,[3 3]);
    %props=regionprops(filt,'Area');
    %maxarea=max([props.Area])/2;
    %blobs=bwareaopen(filt,round(maxarea));
    blobs=filt;%blobs=bwareafilt(filt,5);
    er2=imerode(blobs,strel('disk',2));
    dil2=imdilate(er2,strel('disk',2));
    contourG=activecontour(G,dil2,200);
    
    %Magenta
    adj=imadjust(M);
    compl=imcomplement(adj);
    %threshold=graythresh(compl); %prova threshold assoluto 0.6 e non Otsu
    %bin=imbinarize(compl,threshold);
    bin=compl>190;
    er=imerode(bin,strel('disk',1));
    dil=imdilate(er,strel('disk',1));
    filt=medfilt2(dil,[3 3]);
    %props=regionprops(filt,'Area');
    blobs=filt;%blobs=bwareafilt(filt,5);
    er2=imerode(blobs,strel('disk',2));
    dil2=imdilate(er2,strel('disk',2));
    contourM=activecontour(M,dil2,200);
    %figure,imshow(contourM);
    
    %Lab
    absingle=im2single(ab);
    nColors = 5;
    pixel_labels = imsegkmeans(absingle,nColors,'NumAttempts',3);
    maxlevel=0;
    index=0;
    for i=1:nColors
        mask = pixel_labels==i;
        indices=find(mask);
        graylevels=grayimg(indices);
        meanlevel=mean(graylevels);
        if meanlevel > maxlevel
            maxlevel=meanlevel;
            index=i;
        end
    end
    mask = pixel_labels==index;
    er_mask=imerode(mask,strel('disk',1));
    dil_mask=imdilate(er_mask,strel('disk',1));
    filt_mask=medfilt2(dil_mask,[3 3]);
    contourAB=activecontour(ab,filt_mask);
    
    voting=contourG+contourM+contourAB;
    x=voting>1;
    
    %logical operations
    %xG=(contourG & contourM) | (contourG & contourAB);
    %xM=(contourM & contourG) | (contourM & contourAB);
    %xAB=(contourAB & contourG) | (contourAB & contourM);
    %x = xG | xM | xAB;
    
    xindices=find(x);
    graylevels=grayimg(xindices);
    removegraylevels=graylevels<=190;
    removeindices=find(removegraylevels);
    removexindices=xindices(removeindices);
    x(removexindices)=0;
    
    %xer=imerode(x,strel('disk',2));
    %xdil=imdilate(xer,strel('disk',2));
    %xfilt=medfilt2(xdil,[5 5]);
    %X=bwareaopen(xfilt,400);
    
    bin=x;
    filled = imfill(bin, 'holes');
    holes = filled & ~bin;
    bigholes = bwareaopen(holes, 1000);
    smallholes = holes & ~bigholes;
    filledbin = bin | smallholes;
    er=imerode(filledbin,strel('disk',3));
    dil=imdilate(er,strel('disk',3));
    
    center=round(size(dil)/2+.5);
    r=round(min(size(dil))/2)-(round(min(size(dil))/2)/8);
    rows=size(dil,1);
    cols=size(dil,2);
    centerX=center(2);
    centerY=center(1);
    [columnsInImage, rowsInImage] = meshgrid(1:cols, 1:rows);
    circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= r.^2;
    
    mask=circlePixels | dil;
    biggestarea=bwareafilt(mask,1,4);
    centeredblobs=biggestarea & dil;
    filt=medfilt2(centeredblobs,[2 2]);
    
    imwrite(filt,strcat(selpath,'/',filename,'_bowman_capsule_activecontour_GMab_graylevels2bnocentroids.png'));
end