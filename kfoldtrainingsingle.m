% Script for loading the dataset, filter the features in the PCA space,
% train the ANN with the optimal naumber of neurons in the hidden layer.

load('bowmanActiveContourFullAreaDiameter.mat')
load('datasetidx.mat')
load('Haralick28.mat')
load('mrcLBP120.mat')
%features=features(:,[1 3]);
X=[mrcLBPfeatures haralick28 features];
ScaledDataset=zscore(X);
[coeff,score,latent,tsquared,explained] = pca(ScaledDataset); %PCA sull'intero (anche test) dataset normalizzato con zscore

tokeep=0;
i=1;
while tokeep==0
    if sum(explained(1:i))>=99.9
        tokeep=i;
    end
    i=i+1;
end
sum(explained(1:tokeep)) %prendo le feature che mi mantengono almeno il 99.9 di varianza
ztrainingvalidation=score([Xtrainidx; Xvalididx],1:tokeep);
ztesting=score(Xtestidx,1:tokeep);
%ztrainingvalidation=score([Xtrainidx; Xvalididx],:);
%ztesting=score(Xtestidx,:);
testclasses=classes(Xtestidx,:);
trainingvalidationclasses=classes([Xtrainidx; Xvalididx],:);
png_errori=cell(10);
png_errori_valid=cell(15);
precisionsTest=zeros(10,1);
recallsTest=zeros(10,1);
sspecificitiesTest=zeros(10,1);
accuraciesTest=zeros(10,1);
F1sTest=zeros(10,1);
MCCsTest=zeros(10,1);
h=1;
%pool=parpool;
%rng('default');
for j=1:10
    net = patternnet(27,'trainscg'); %definisco patternnet
    %net.layers{1}.initFcn = 'initwb';
    %net.layers{2}.initFcn = 'initwb';
    %net.inputWeights{1,1}.initFcn = 'randsmall';
    %net.layerWeights{2,1}.initFcn= 'randsmall';
    %net.biases{1,1}.initFcn='randsmall';
    %net.biases{2,1}.initFcn='randsmall';
    %rng('default');
    parts=cvpartition(trainingvalidationclasses,'kfold',10); % kfold per dividere training e validation set
    numNN = 10; %numero di reti con diverse inizializzazioni per ogni fold
    NN = cell(numNN,parts.NumTestSets);
    confusionMat = cell(numNN,parts.NumTestSets);
    perfs=zeros(numNN,parts.NumTestSets);
    threshold=zeros(numNN,parts.NumTestSets);
    precisionsValid=zeros(numNN,parts.NumTestSets);
    recallsValid=zeros(numNN,parts.NumTestSets);
    specificitiesValid=zeros(numNN,parts.NumTestSets);
    accuraciesValid=zeros(numNN,parts.NumTestSets);
    F1sValid=zeros(numNN,parts.NumTestSets);
    MCCsValid=zeros(numNN,parts.NumTestSets);
    crossentropyValid=zeros(numNN,parts.NumTestSets);
    crossentropyTrain=zeros(numNN,parts.NumTestSets);
    for k=1:parts.NumTestSets %ciclo per ogni fold
        trainInd=find(parts.training(k));
        valInd=find(parts.test(k));
        [trainInd,valInd,testInd] = divideind(size(trainingvalidationclasses,1),trainInd,valInd);
        validationclasses=trainingvalidationclasses(valInd,:);
        zvalidation=ztrainingvalidation(valInd,1:tokeep);
        %zvalidation=ztrainingvalidation(valInd,:);
        net.divideFcn='divideind';
        net.divideParam.trainInd=trainInd;
        net.divideParam.valInd=valInd;
        net.divideParam.testInd=testInd;
        %net.trainParam.showWindow = false;
        net.performFcn = 'crossentropy';
        net.trainParam.max_fail = 6;
        %rng(j);
        for i = 1:numNN %ciclo per ogni rete della fold
            net=init(net);
            %wb(:,i)=getwb(net);
            fprintf(strcat(num2str(j),' iteration ---',num2str(k),' fold: Training ', num2str(i),'/', num2str(numNN),'\n'));
            %trainingvalidationclassesweights=zeros(size(trainingvalidationclasses));
            %trainingvalidationclassesweights(find(trainingvalidationclasses==1))=0.8;
            %trainingvalidationclassesweights(find(trainingvalidationclasses==0))=0.2;
            %[NN{i,k}, tr] = train(net, ztrainingvalidation', trainingvalidationclasses',{},{},trainingvalidationclassesweights'); %training della i-esima rete della k-esima fold
            [NN{i,k}, tr] = train(net, ztrainingvalidation', trainingvalidationclasses'); %training della i-esima rete della k-esima fold
            yvalid = NN{i,k}(zvalidation'); %output sul validation test della i-esima rete della k-esima fold
            crossentropyValid(i,k) = tr.best_vperf; %mse(net, validationclasses, yvalid);
            crossentropyTrain(i,k) = tr.best_perf;
            [Xroc,Yroc,T,AUC,OPTROCPT] = perfcurve(validationclasses,yvalid',1); %ROC curve della i-esima rete della k-esima fold
            threshold(i,k)=T((Xroc==OPTROCPT(1))&(Yroc==OPTROCPT(2))); %threshold migliore sulla base del punto ottimale OPTROCPT
            
            distance=sqrt((1-Yroc).^2 + Xroc.^2);
            [mdist,mdistidx]=min(distance);
            thresholddist(i,k)=T(mdistidx);
            
            yvalidround=yvalid>=threshold(i,k); %classificazione in base a threshold migliore
            confmat=confusionmat(validationclasses,double(yvalidround));
            confusionMat{i,k}=confmat;
            [precision, recall, specificity, accuracy, F1, MCC]=computeConfusionMetrics(confmat); %metriche valutate sulla classificazione del validation test
            precisionsValid(i,k)=precision;
            recallsValid(i,k)=recall;
            specificitiesValid(i,k)=specificity;
            accuraciesValid(i,k)=accuracy;
            F1sValid(i,k)=F1;
            MCCsValid(i,k)=MCC;
            
            yvalidrounddist=yvalid>=thresholddist(i,k); %classificazione in base a threshold migliore
            confmatdist=confusionmat(validationclasses,double(yvalidrounddist));
            confusionMatdist{i,k}=confmatdist;
            [precisiondist, recalldist, specificitydist, accuracydist, F1dist, MCCdist]=computeConfusionMetrics(confmatdist); %metriche valutate sulla classificazione del validation test
            precisionsValiddist(i,k)=precisiondist;
            recallsValiddist(i,k)=recalldist;
            specificitiesValiddist(i,k)=specificitydist;
            accuraciesValiddist(i,k)=accuracydist;
            F1sValiddist(i,k)=F1dist;
            MCCsValiddist(i,k)=MCCdist;
        end
    end
    [M,idx]=max(MCCsValid); %identifico reti migliori per ogni fold sulla base di una metrica
    [Mdist,idxdist]=max(MCCsValiddist);
    
    bestnet=cell(1,length(idx));
    bestthreshold=zeros(1,length(idx));
    precisionsBestValid=zeros(1,length(idx));
    recallsBestValid=zeros(1,length(idx));
    specificitiesBestValid=zeros(1,length(idx));
    accuraciesBestValid=zeros(1,length(idx));
    F1sBestValid=zeros(1,length(idx));
    MCCsBestValid=zeros(1,length(idx));
    crossentropyBestValid=zeros(1,length(idx));
    crossentropyBestTrain=zeros(1,length(idx));
    ybestvalid=zeros(size(zvalidation,1),length(idx));
    ytest=zeros(size(ztesting,1),length(idx));
    ytestclassesbestnet=zeros(size(ztesting,1),length(idx));
    %calcolo output sul test set per tutte le k migliori reti con il relativo threshold
    for i=1:length(idx)
        bestnet{i}=NN{idx(i),i};
        bestthreshold(i)=threshold(idx(i),i);
        ybestvalid(:,i)=bestnet{i}(zvalidation');
        precisionsBestValid(i)=precisionsValid(idx(i),i);
        recallsBestValid(i)=recallsValid(idx(i),i);
        specificitiesBestValid(i)=specificitiesValid(idx(i),i);
        accuraciesBestValid(i)=accuraciesValid(idx(i),i);
        F1sBestValid(i)=F1sValid(idx(i),i);
        MCCsBestValid(i)=MCCsValid(idx(i),i);
        crossentropyBestValid(i)=crossentropyValid(idx(i),i);
        crossentropyBestTrain(i)=crossentropyTrain(idx(i),i);
        ytest(:,i)=bestnet{i}(ztesting');
        ytestclassesbestnet(:,i)=ytest(:,i)>=bestthreshold(i);
    end
    ytestclassesbestnetsum=sum(ytestclassesbestnet,2);
    
    for i=1:length(idxdist)
        bestnet{i}=NN{idxdist(i),i};
        bestthresholddist(i)=thresholddist(idxdist(i),i);
        ybestvaliddist(:,i)=bestnet{i}(zvalidation');
        precisionsBestValiddist(i)=precisionsValiddist(idxdist(i),i);
        recallsBestValiddist(i)=recallsValiddist(idxdist(i),i);
        specificitiesBestValiddist(i)=specificitiesValiddist(idxdist(i),i);
        accuraciesBestValiddist(i)=accuraciesValiddist(idxdist(i),i);
        F1sBestValiddist(i)=F1sValiddist(idxdist(i),i);
        MCCsBestValiddist(i)=MCCsValiddist(idxdist(i),i);
        %crossentropyBestValiddist(i)=crossentropyValiddist(idxdist(i),i);
        %crossentropyBestTraindist(i)=crossentropyTraindist(idxdist(i),i);
        ytestdist(:,i)=bestnet{i}(ztesting');
        ytestclassesbestnetdist(:,i)=ytestdist(:,i)>=bestthresholddist(i);
    end
    ytestclassesbestnetsumdist(:,j)=sum(ytestclassesbestnetdist,2);
    
    %%%%%%%%
    %normalizzazione min-max a somma 1 della metrica delle migliori reti per ogni fold per utilizzarle come pesi nel voting
    %mm=minmax(M);
    %zM=(M-mm(1))/(mm(2)-mm(1));
    %zM=M/sum(M);
    %sortedzM=sort(zM,'descend');
    %WMVthreshold1=sum(sortedzM(1:5));
    %%%%%%%%
    %Weighted Majority Voting con pesi calcolati precedentemente dei valori predetti sul test set da ciascuna k migliore rete
    %WMV1=ytestclassesbestnetsum*zM';
    %WMV1=ytest*zM';
    
    %WMVthreshold1=bestthreshold*zM';
    %errors=testclasses-WMV1;
    ytestclasses1=ytestclassesbestnetsum>=5;
    figure,plotconfusion(testclasses',ytestclasses1'),title('ROC OPT');
    
    ytestclasses1dist=ytestclassesbestnetsumdist(:,j)>=5;
    figure,plotconfusion(testclasses',ytestclasses1dist'),title('distance');
    
    testconf1=confusionmat(testclasses',double(ytestclasses1'));
    [precision1,recall1,specificity1,accuracy1,F11,MCC1]=computeConfusionMetrics(testconf1);
    confr1=xor(ytestclasses1,testclasses);
    confridx1=find(confr1);
    png_errori{h}=png_files(confridx1);
    precisionsTest(h)=precision1;
    recallsTest(h)=recall1;
    sspecificitiesTest(h)=specificity1;
    accuraciesTest(h)=accuracy1;
    F1sTest(h)=F11;
    MCCsTest(h)=MCC1;
    
    testconf1dist=confusionmat(testclasses',double(ytestclasses1dist'));
    [precision1dist,recall1dist,specificity1dist,accuracy1dist,F11dist,MCC1dist]=computeConfusionMetrics(testconf1dist);
    confr1dist=xor(ytestclasses1dist,testclasses);
    confridx1dist=find(confr1dist);
    png_erroridist{h}=png_files(confridx1dist);
    precisionsTestdist(h)=precision1dist;
    recallsTestdist(h)=recall1dist;
    sspecificitiesTestdist(h)=specificity1dist;
    accuraciesTestdist(h)=accuracy1dist;
    F1sTestdist(h)=F11dist;
    MCCsTestdist(h)=MCC1dist;
    
    meanprecisionsBestValid(h)=mean(precisionsBestValid);
    stdprecisionsBestValid(h)=std(precisionsBestValid);
    meanrecallsBestValid(h)=mean(recallsBestValid);
    stdrecallsBestValid(h)=std(recallsBestValid);
    meanspecificitiesBestValid(h)=mean(specificitiesBestValid);
    stdspecificitiesBestValid(h)=std(specificitiesBestValid);
    meanaccuraciesBestValid(h)=mean(accuraciesBestValid);
    stdaccuraciesBestValid(h)=std(accuraciesBestValid);
    meanF1sBestValid(h)=mean(F1sBestValid);
    stdF1sBestValid(h)=std(F1sBestValid);
    meanMCCsBestValid(h)=mean(MCCsBestValid);
    stdMCCsBestValid(h)=std(MCCsBestValid);
    meancrossentropyBestValid(h)=mean(crossentropyBestValid);
    meancrossentropyBestTrain(h)=mean(crossentropyBestTrain);
    h=h+1;
    
    meanprecisionsValid(h)=mean(mean(precisionsValid));
    meanrecallsValid(h)=mean(mean(recallsValid));
    meanspecificitiesValid(h)=mean(mean(specificitiesValid));
    meanaccuraciesValid(h)=mean(mean(accuraciesValid));
    meanF1sValid(h)=mean(mean(F1sValid));
    meanMCCsValid(h)=mean(mean(MCCsValid));
    
    meanprecisionsValiddist(h)=mean(mean(precisionsValiddist));
    meanrecallsValiddist(h)=mean(mean(recallsValiddist));
    meanspecificitiesValiddist(h)=mean(mean(specificitiesValiddist));
    meanaccuraciesValiddist(h)=mean(mean(accuraciesValiddist));
    meanF1sValiddist(h)=mean(mean(F1sValiddist));
    meanMCCsValiddist(h)=mean(mean(MCCsValiddist));
end