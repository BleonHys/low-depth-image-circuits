%% Plots for ZHAW SoE - Quantum Computing HS25 - QML-Project
% init
clc
clear

% constants

no_trainEpochs = 150;
no_fold = 5;

%% Import data
% import and read sampleweighted accuracy data for from .csv


col_maj = zeros(no_trainEpochs,no_fold);
col_maj(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-column_major__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
col_maj(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-column_major__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
col_maj(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-column_major__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
col_maj(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-column_major__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
col_maj(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-column_major__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

diag = zeros(no_trainEpochs,no_fold);
diag(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
diag(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
diag(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
diag(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
diag(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

row_maj = zeros(no_trainEpochs,no_fold-1);
row_maj(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-row_major__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
%row_maj(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-row_major__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
row_maj(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-row_major__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
row_maj(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-row_major__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
row_maj(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-row_major__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

diag_zigzag = zeros(no_trainEpochs,no_fold);
diag_zigzag(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal_zigzag__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
diag_zigzag(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal_zigzag__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
diag_zigzag(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal_zigzag__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
diag_zigzag(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal_zigzag__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
diag_zigzag(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-diagonal_zigzag__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

cor_spiral = zeros(no_trainEpochs,no_fold);
cor_spiral(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-corner_spiral__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
cor_spiral(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-corner_spiral__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
cor_spiral(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-corner_spiral__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
cor_spiral(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-corner_spiral__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
cor_spiral(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-corner_spiral__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

skake = zeros(no_trainEpochs,no_fold);
snake(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-snake__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
snake(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-snake__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
snake(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-snake__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
snake(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-snake__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
snake(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-snake__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

snake_vert = zeros(no_trainEpochs,no_fold);
snake_vert(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-vertical_snake__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
snake_vert(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-vertical_snake__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
snake_vert(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-vertical_snake__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
snake_vert(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-vertical_snake__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
snake_vert(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-vertical_snake__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

hilb = zeros(no_trainEpochs,no_fold);
hilb(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-hilbert__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
hilb(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-hilbert__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
hilb(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-hilbert__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
hilb(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-hilbert__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
hilb(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-hilbert__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

mort = zeros(no_trainEpochs,no_fold-1);
mort(:,1) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-morton__k2000__p1__s42\vqc_linear\fold0\seed42\training_data_epoch.csv.").data(:,6);
mort(:,2) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-morton__k2000__p1__s42\vqc_linear\fold1\seed42\training_data_epoch.csv.").data(:,6);
mort(:,3) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-morton__k2000__p1__s42\vqc_linear\fold2\seed42\training_data_epoch.csv.").data(:,6);
mort(:,4) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-morton__k2000__p1__s42\vqc_linear\fold3\seed42\training_data_epoch.csv.").data(:,6);
%mort(:,5) = importdata("..\results\mnist_encoding_ablation_paperlike\mnist__idx-morton__k2000__p1__s42\vqc_linear\fold4\seed42\training_data_epoch.csv.").data(:,6);

%% calc
% calculate average and standard deviation

% xyz_data = array[avg,std]
col_maj_data = zeros(no_trainEpochs,2);
diag_data = zeros(no_trainEpochs,2);
row_maj_data = zeros(no_trainEpochs,2);
diag_zigzag_data = zeros(no_trainEpochs,2);
cor_spiral_data = zeros(no_trainEpochs,2);
snake_data = zeros(no_trainEpochs,2);
snake_vert_data = zeros(no_trainEpochs,2);
hilb_data = zeros(no_trainEpochs,2);
mort_data = zeros(no_trainEpochs,2);

%calc mean and std of all matrix rows
col_maj_data(:,1) = mean(col_maj*100,2);
diag_data(:,1) = mean(diag*100,2);
row_maj_data(:,1) = mean(row_maj*100,2);
diag_zigzag_data(:,1) = mean(diag_zigzag*100,2);
cor_spiral_data(:,1) = mean(cor_spiral*100,2);
snake_data(:,1) = mean(snake*100,2);
snake_vert_data(:,1) = mean(snake_vert*100,2);
hilb_data(:,1) = mean(hilb*100,2);
mort_data(:,1) = mean(mort*100,2);

col_maj_data(:,2) = std(col_maj*100,0,2);
diag_data(:,2) = std(diag*100,0,2);
row_maj_data(:,2) = std(row_maj*100,0,2);
diag_zigzag_data(:,2) = std(diag_zigzag*100,0,2);
cor_spiral_data(:,2) = std(cor_spiral*100,0,2);
snake_data(:,2) = std(snake*100,0,2);
snake_vert_data(:,2) = std(snake_vert*100,0,2);
hilb_data(:,2) = std(hilb*100,0,2);
mort_data(:,2) = std(mort*100,0,2);


%% Plot graphs 
% with error bars
figure(1)
orient landscape
eb1 = errorbar((1:no_trainEpochs),col_maj_data(:,1),col_maj_data(:,2)/100,'LineWidth',1);
hold on
eb2 = errorbar((1:no_trainEpochs),diag_data(:,1),diag_data(:,2)/100,'LineWidth',1);
eb3 = errorbar((1:no_trainEpochs),row_maj_data(:,1),row_maj_data(:,2)/100,'LineWidth',1);
eb4 = errorbar((1:no_trainEpochs),diag_zigzag_data(:,1),diag_zigzag_data(:,2)/100,'LineWidth',1);
eb5 = errorbar((1:no_trainEpochs),cor_spiral_data(:,1),cor_spiral_data(:,2)/100,'LineWidth',1);
eb6 = errorbar((1:no_trainEpochs),snake_data(:,1),snake_data(:,1)/100,'LineWidth',1);
eb7 = errorbar((1:no_trainEpochs),snake_vert_data(:,1),snake_vert_data(:,2)/100,'LineWidth',1);
eb8 = errorbar((1:no_trainEpochs),hilb_data(:,1),hilb_data(:,2)/100,'LineWidth',1,'LineStyle','--');
eb9 = errorbar((1:no_trainEpochs),mort_data(:,1),mort_data(:,2)/100,'LineWidth',1,'LineStyle','--');
hold off
title('MNIST classification accuracy with different image encoding');
legend('Column Major','Left Diagonal','Row Major','Zig-Zag Diagonal','Corner Spiral','Horizontal Snake','Vertical Snake','Hilbert','Morton','Location','northwest');
xlabel('No. of training epochs');
ylabel('classification accuracy in %');
ylim([0, 50]);
grid on;

% without errorbars
figure(2)
orient landscape
plt1 = plot((1:no_trainEpochs),col_maj_data(:,1),'LineWidth',1);
hold on
plt2 = plot((1:no_trainEpochs),diag_data(:,1),'LineWidth',1);
plt3 = plot((1:no_trainEpochs),row_maj_data(:,1),'LineWidth',1);
plt4 = plot((1:no_trainEpochs),diag_zigzag_data(:,1),'LineWidth',1);
plt5 = plot((1:no_trainEpochs),cor_spiral_data(:,1),'LineWidth',1);
plt6 = plot((1:no_trainEpochs),snake_data(:,1),'LineWidth',1);
plt7 = plot((1:no_trainEpochs),snake_vert_data(:,1),'LineWidth',1);
plt8 = plot((1:no_trainEpochs),hilb_data(:,1),'LineWidth',1,'LineStyle','--');
plt9 = plot((1:no_trainEpochs),mort_data(:,1),'LineWidth',1,'LineStyle','--');
hold off
title('MNIST classification accuracy with different image encoding');
legend('Column Major','Left Diagonal','Row Major','Zig-Zag Diagonal','Corner Spiral','Horizontal Snake','Vertical Snake','Hilbert','Morton','Location','northwest');
xlabel('No. of training epochs');
ylabel('classification accuracy in %');
ylim([0, 50]);
grid on;

%print('-f1','QMLProject_ImageEncoding_Performance_Errorbar','-dpng');
%print('-f2','QMLProject_ImageEncoding_Performance','-dpng');

