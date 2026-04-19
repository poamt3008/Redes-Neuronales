% Este programa estima los parámetros a y b de una recta (y = ax + b)
% a partir de datos con ruido, utilizando el método de mínimos cuadrados
% (solución exacta matricial).

%********************* Inicialización ************************************
clc
clear all
close all

%********************* Generación de datos *******************************
a = 3;          % Pendiente real
b = -6;         % Intercepto real
noise_A = 1;    % Amplitud del ruido gaussiano

limInf = -10;   
limSup = 10;
fs = 10;        % Frecuencia de muestreo

x = limInf:1/fs:limSup;
x = x';         % Vector columna
nx = length(x); % Número de muestras

% Señal con ruido
y = a*x + b + noise_A*randn(nx,1);

%********************* Método exacto (mínimos cuadrados) *****************
% Construcción del sistema normal A*z = B

A11 = sum(x.*x);
A12 = sum(x);
A21 = sum(x);
A22 = nx;

B1 = sum(x.*y);
B2 = sum(y);

A = [A11 A12
     A21 A22];

B = [B1
     B2];

% Solución del sistema
z = A\B;

an = z(1,1);    % Pendiente estimada
bn = z(2,1);    % Intercepto estimado

% Mostrar resultados
disp("Datos Reales vs Datos Estimados")
[a an
 b bn]

% Error porcentual
error_a = abs((a-an)*100/a)
error_b = abs((b-bn)*100/b)

%*********************** Gráfica  ********************************

% ---- Parámetros de tamaño (en cm) ----
fig_width = 12;   % ancho en cm
fig_height = 8;   % alto en cm

figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 fig_width fig_height]);

% ---- Datos ----
plot(x, y, 'o', 'MarkerSize', 4, 'DisplayName','Datos con ruido')
grid on; grid minor;
hold on;

% ---- Recta estimada ----
y_est = an*x + bn;
plot(x, y_est, 'LineWidth', 2, 'DisplayName','Recta estimada')

% ---- Etiquetas ----
xlabel('x')
ylabel('y')
title('Regresión Lineal por Mínimos Cuadrados')

% ---- Leyenda ----
legend('Location','best')

% ---- Texto dinámico ----
x_pos = min(x) + 0.05*(max(x)-min(x));
y_pos = max(y) - 0.1*(max(y)-min(y));

txt = sprintf(['Real: a=%.2f, b=%.2f\n' ...
               'Estimado: a=%.2f, b=%.2f\n' ...
               'Error: a=%.2f%%, b=%.2f%%'], ...
               a, b, an, bn, error_a, error_b);

text(x_pos, y_pos, txt, ...
    'FontSize',9, ...
    'BackgroundColor','w', ...
    'EdgeColor','k');

hold off;
