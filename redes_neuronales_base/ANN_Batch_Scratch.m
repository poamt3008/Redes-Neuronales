%************************************************************************
    % Entrenamiento de RNs por metodo batch
    %************************************************************************
    
    clc
    clear
    close all
    
    %************************************************************************
    % Generación de data set
    %************************************************************************
    
    %*********************Parámetros reales**********************************
    a = 3;
    b = -5.5;
    
    %*********************Nivel de Ruido**************************************
    noise_A = 1;
    
    %*********************Datos base*****************************************
    limInf = -2;
    limSup = 3;
    fs = 30;
    
    x = (limInf:1/fs:limSup)';
    nx = length(x);
    
    y_real = a*x + b + noise_A*randn(nx,1); % Data con ruido
    
    %************************************************************************
    % Configuracion de Red Neuronal
    %************************************************************************
    
    ne = 1;     % Número de entradas
    nm = 20;    % Número de neuronas intermedias
    ns = 1;     % Número de salidas
    
    eta = 0.01;         % Razon de aprendizaje
    errorporc = 0.0075; % Criterio de convergencia (%)
    Jold = inf;         % Inicialización de costo
    
    %********************* Activar Bias *************************************
    neuronaBias = 1;
    
    if neuronaBias == 1
        x = [x ones(nx,1)];
    end
    
    ne = size(x,2);     % Ajuste automático de entradas
    
    %********************* Inicialización ***********************************
    v = 0.05*randn(ne,nm);
    w = 0.05*randn(nm,ns);
    
    % Prealocación
    y = zeros(nx,ns);
    err_vec = zeros(nx,ns);
    J = zeros(50000,1);
    
    %************************************************************************
    % Algoritmo de entrenamiento - Metodo Batch
    %************************************************************************
    
    for iter = 1:50000
        
        err_vec = zeros(nx,ns);   % Reiniciar error por iteración
    
        dJdw = zeros(nm,ns);
        dJdv = zeros(ne,nm);
    
        for k = 1:nx
            
            in = x(k,:)';
            
            %-------------------- Forward --------------------
            m = v' * in;
            n = 1.0 ./ (1 + exp(-m));     % Activación
            out = w' * n;
            
            y(k,:) = out';
            
            %-------------------- Error ----------------------
            er = out - y_real(k,:)';
            err_vec(k,:) = er';
            
            %-------------------- Backprop -------------------
            dndm = n .* (1 - n);
            
            % Gradientes modificado para batch - acumula
            dJdw = dJdw + n * er';                     
            dJdv = dJdv + in * ((w * er) .* dndm)';
            
                   
        end
        % Update BATCH
        w = w - eta * dJdw/nx;
        v = v - eta * dJdv/nx;
        
        %-------------------- Costo --------------------------
        JJ = 0.5 * sum(sum(err_vec.^2)) / nx;
        
        %-------------------- Convergencia -------------------
        dJpor = abs((JJ - Jold) / JJ) * 100;
        J(iter) = JJ;
        Jold = JJ;
        
        if dJpor < errorporc
            break;
        end    
        
    end
    
    % Recorte de vector J
    J = J(1:iter);
    
    %************************************************************************
    % Gráficas
    %************************************************************************
    
    figure(1)
    plot(J,'LineWidth',1.5)
    grid on
    grid minor
    title(sprintf('Función de costo - Iter=%d',iter))
    xlabel('Iteración')
    ylabel('J')
    
    figure(2)
    x_plot = x(:,1);
    
    plot(x_plot, y, 'LineWidth',1.5)
    hold on
    plot(x_plot, y_real, '*')
    legend('Red','Datos','Location','best')
    grid on
    grid minor
    title(sprintf('Ajuste de la Red (Bias=%d)',neuronaBias))
    xlabel('x')
    ylabel('y')
