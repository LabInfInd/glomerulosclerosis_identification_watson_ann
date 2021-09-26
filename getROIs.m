% Script for creating the knowledge base: staring from svs files, positive
% and negative folders with samples are created based on the annotations
% contained in the original file.

selpath=uigetdir;
svs_files=dir(fullfile(selpath,'*.svs'));
%xml_files=dir(fullfile(selpath,'*.xml'));
positive_dataset=0;
negative_dataset=0;
for svs_num=1:size(svs_files)
    [filepath,filename,ext]=fileparts(svs_files(svs_num).name);
    if isfile(strcat(selpath,'/',filename,'.xml'))
        svs_file=strcat(selpath,'/',filename,'.svs');
        xml_file=strcat(selpath,'/',filename,'.xml');
        disp(strcat('XML found for ',svs_files(svs_num).name,'. Extracting annotated (positive and negative) ROIs...'));
        svs=imread(svs_file);
        positive_num=0;
        negative_num=0;
        ignored_num=0;
        
        xDoc = xmlread(xml_file);
        annotations=xDoc.getElementsByTagName('Annotation');
        for annot_index=0:annotations.getLength-1
            annotation=annotations.item(annot_index);
            attributes=annotation.getElementsByTagName('Attribute');
            for attr_index=0:attributes.getLength-1
                attribute=attributes.item(attr_index);
                atts=attribute.getAttributes;
                for atts_index=0:atts.getLength-1
                    att=atts.item(atts_index);
                    if strcmpi(att.getName,'Value')
                        if (strcmpi(att.getValue, 'negativo')) || (strcmpi(att.getValue, 'negative'))
                            if ~exist(strcat(selpath,'/Negativi/'),'dir')
                                mkdir(strcat(selpath,'/Negativi'));
                            end
                            directory=strcat(selpath,'/Negativi/');
                            label='Negativo';
                        elseif (strcmpi(att.getValue, 'positivo')) || (strcmpi(att.getValue, 'positive'))
                            if ~exist(strcat(selpath,'/Positivi/'),'dir')
                                mkdir(strcat(selpath,'/Positivi'));
                            end
                            directory=strcat(selpath,'/Positivi/');
                            label='Positivo';
                        else
                            directory=[];
                        end
                    end
                end
            end
            
            if ~isempty(directory)
                Regions=annotation.getElementsByTagName('Region'); % get a list of all the region tags
                if strcmp(directory,strcat(selpath,'/Positivi/'))
                    positive_num=Regions.getLength;
                    positive_dataset=positive_dataset+positive_num;
                else
                    negative_num=Regions.getLength;
                    negative_dataset=negative_dataset+negative_num;
                end
                for regioni = 0:Regions.getLength-1
                    Region=Regions.item(regioni);  % for each region tag
                    
                    %get a list of all the vertexes (which are in order)
                    verticies=Region.getElementsByTagName('Vertex');
                    xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
                    for vertexi = 0:verticies.getLength-1 %iterate through all verticies
                        %get the x value of that vertex
                        x=str2double(verticies.item(vertexi).getAttribute('X'));
                        
                        %get the y value of that vertex
                        y=str2double(verticies.item(vertexi).getAttribute('Y'));
                        xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
                    end
                    
                    %extract annotation x and y vertices & create mask
                    xi=xy{regioni+1}(:,1);
                    yi=xy{regioni+1}(:,2);
                    mask=poly2mask(xi,yi,size(svs,1),size(svs,2));
                    
                    %ignore small bounding box
                    props=regionprops(mask,'Area','BoundingBox');
                    index=0;
                    area=0;
                    for i=1:numel(props)
                        if props(i).Area >= area
                            area=props(i).Area;
                            index=i;
                        end
                    end
                    
                    %crop & save ROI
                    bb=props(index).BoundingBox;
                    bb(1)=bb(1)-round(bb(3)*0.05); % -5% width starting ul-point
                    bb(2)=bb(2)-round(bb(4)*0.05);% -5% height starting ul-point
                    bb(3)=bb(3)+(2*round(bb(3)*0.05)); % +5% width
                    bb(4)=bb(4)+(2*round(bb(4)*0.05)); % +5% height
                    glomerulo=imcrop(svs,bb);
                    glomerulo_filename=strcat(directory, filename,'_glomerulo_',label,'_',num2str(regioni+1),'.png');
                    imwrite(glomerulo,glomerulo_filename);
                end
            else
                Regions=annotation.getElementsByTagName('Region'); % get a list of all the region tags
                ignored_num=Regions.getLength;
            end
        end
        disp(strcat('Annotated ROIs successfully saved! ',num2str(positive_num),' Positive, ',num2str(negative_num),' Negative, ',num2str(ignored_num),' Ignored.'));
    else
        disp(strcat('No XML found for ',svs_files(svs_num).name,'...'));
    end
end
disp(strcat('Operation completed! Total dataset: ',num2str(positive_dataset),' Positive, ',num2str(negative_dataset),' Negative.'));
