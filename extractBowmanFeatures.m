% Script for creating features from the bowman capsule.

clear
clc
Xtrainidx=[];
Xtestidx=[];
Xvalididx=[];
selpath=uigetdir;
bowman_png_files=dir(fullfile(selpath,'*_capsule*.png'));
for png_num=1:size(bowman_png_files)
    [filepath,filename,ext]=fileparts(bowman_png_files(png_num).name);
    bowmanmask=imread(strcat(selpath,'/',bowman_png_files(png_num).name));
    props=regionprops(bowmanmask,'Area');
    %num=numel(props); %Numero di elementi
    area=sum([props.Area]);
    
    center=round(size(bowmanmask)/2+.5);
    r=round(min(size(bowmanmask))/2)-(round(min(size(bowmanmask))/2)/3);
    rows=size(bowmanmask,1);
    cols=size(bowmanmask,2);
    centerX=center(2);
    centerY=center(1);
    [columnsInImage, rowsInImage] = meshgrid(1:cols, 1:rows);
    circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= r.^2;
    centeredblobs=circlePixels & bowmanmask;
    centeredprops=regionprops(centeredblobs,'Area');
    centeredarea=sum([centeredprops.Area]);
    
    totalimagearea=rows*cols;
    arearatio=area/totalimagearea;
    centeredarearatio=centeredarea/totalimagearea;
    
    convhull=bwconvhull(bowmanmask);
    hullprops=regionprops(convhull,'Area','Perimeter','EquivDiameter','MajorAxisLength','MinorAxisLength');
    if numel(hullprops)~=0
        %AR = [hullprops.MajorAxisLength]./[hullprops.MinorAxisLength];
        %circularity = ((4*pi*[hullprops.Area]) ./ ([hullprops.Perimeter]).^2);
        diameter= [hullprops.EquivDiameter];
    else
        %AR = 0;
        %circularity = 0;
        diameter = 0;
    end
    if contains(filename,'negativo','IgnoreCase',true)
        classes(png_num)=0;
    else
        classes(png_num)=1;
    end
    if contains(filename,'training','IgnoreCase',true)
        Xtrainidx=[Xtrainidx ; png_num];
    elseif contains(filename,'testing','IgnoreCase',true)
        Xtestidx=[Xtestidx ; png_num];
    else
        Xvalididx=[Xvalididx ; png_num];
    end
    %features(png_num,:)=[arearatio, centeredarearatio, diameter];
    features(png_num,:)=[arearatio, diameter];
end
savepath=strcat(selpath,'/',filename,'_features.mat');
save(savepath,'features');
classes=classes';
savepath=strcat(selpath,'/',filename,'_classes.mat');
save(savepath,'classes');