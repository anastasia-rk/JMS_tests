clc; clear;
%% figures
close all;
set(0,'DefaultFigureWindowStyle','docked');
set(0,'defaultFigureColor',[1 1 1]);
set(0,'defaultTextInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultColorbarTickLabelInterpreter','latex');
set(0,'defaultAxesTitleFontWeight','normal');
set(0,'defaultAxesFontSize',16);

%% Custom colormap
my_map = cutsom_colormap(256);
