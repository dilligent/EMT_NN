% =========================================================================
% MATLAB script to batch process COMSOL simulations for effective thermal
% conductivity calculation from JSON geometry files.
% =========================================================================
clear; clc; close all;

% Import COMSOL Java model
import com.comsol.model.*
import com.comsol.model.util.*

%% --- Configuration ---
% Directory containing the JSON files
jsonDir = 'json_files'; 
% Output file for results
outputCsvFile = 'effective_conductivity_results.csv';

% Simulation Parameters
T_hot = 373.15;  % K (100 C)
T_cold = 273.15; % K (0 C)
deltaT = T_hot - T_cold;

%% --- Main Processing Loop ---

% Find all JSON files in the directory
jsonFiles = dir(fullfile(jsonDir, '*.json'));
numFiles = length(jsonFiles);
fprintf('Found %d JSON files to process.\n', numFiles);

% Initialize results table
results_data = cell(numFiles, 6); % sample_id, kxx, kxy, kyx, kyy, phi

% Start connection to COMSOL Server (make sure it's running!)
model = ModelUtil.connect('localhost');
fprintf('Successfully connected to COMSOL server.\n\n');


for i = 1:numFiles
    % Create a new model component for each file
    model.modelNode.create('mod1');
    model.component.create('comp1', true);
    
    jsonPath = fullfile(jsonFiles(i).folder, jsonFiles(i).name);
    fprintf('Processing file %d/%d: %s\n', i, numFiles, jsonFiles(i).name);
    
    try
        % --- 1. Load and Parse JSON ---
        jsonText = fileread(jsonPath);
        data = jsondecode(jsonText);
        
        % Extract metadata
        meta = data.meta;
        Lx = meta.Lx;
        Ly = meta.Ly;
        km = meta.km;
        ki = meta.ki;
        
        sample_id = string(meta.sample_id);
        phi_actual = meta.phi;
        
        
        % --- 2. Build Geometry in COMSOL ---
        geom = model.component('comp1').geom.create('geom1', 2);
        
        % Create the main rectangle (matrix)
        matrixRect = geom.create('r1', 'Rectangle');
        matrixRect.set('size', [Lx, Ly]);
        
        % Create ellipse objects for inclusions
        ellipses_geom = {};
        for j = 1:length(data.ellipses)
            ell = data.ellipses(j);
            ellTagName = ['ell' num2str(j)];
            ellipses_geom{j} = geom.create(ellTagName, 'Ellipse');
            ellipses_geom{j}.set('pos', [ell.x, ell.y]);
            ellipses_geom{j}.set('semiaxes', [ell.a, ell.b]);
            ellipses_geom{j}.set('rot', ell.theta_deg);
        end
        
        % Form the composite by subtracting ellipses from the rectangle
        diffOp = geom.create('dif1', 'Difference');
        diffOp.selection('input').set('r1'); % Object to add
        diffOp.selection('input2').set(cellfun(@(c) c.tag, ellipses_geom, 'UniformOutput', false)); % Objects to subtract
        
        geom.run;
        
        % --- 3. Create Selections for Boundaries (Robust Method) ---
        sel_b_left = geom.create('sel_b_left', 'BoxSelection');
        sel_b_left.set('xmin', -0.1*Lx); sel_b_left.set('xmax', 0);
        sel_b_left.set('entitydim', 1); % Select boundaries
        
        sel_b_right = geom.create('sel_b_right', 'BoxSelection');
        sel_b_right.set('xmin', Lx); sel_b_right.set('xmax', 1.1*Lx);
        sel_b_right.set('entitydim', 1);
        
        sel_b_bottom = geom.create('sel_b_bottom', 'BoxSelection');
        sel_b_bottom.set('ymin', -0.1*Ly); sel_b_bottom.set('ymax', 0);
        sel_b_bottom.set('entitydim', 1);
        
        sel_b_top = geom.create('sel_b_top', 'BoxSelection');
        sel_b_top.set('ymin', Ly); sel_b_top.set('ymax', 1.1*Ly);
        sel_b_top.set('entitydim', 1);
        
        geom.run; % Run geom again to finalize selections
        
        
        % --- 4. Define Materials ---
        mat_matrix = model.component('comp1').material.create('mat1', 'Common');
        mat_matrix.label('Matrix');
        mat_matrix.propertyGroup('def').set('thermalconductivity', {'km' '0' '0'; '0' 'km' '0'; '0' '0' 'km'});
        mat_matrix.selection.set(1); % Selection for the matrix domain (usually domain 1)
        
        mat_inclusion = model.component('comp1').material.create('mat2', 'Common');
        mat_inclusion.label('Inclusion');
        mat_inclusion.propertyGroup('def').set('thermalconductivity', {'ki' '0' '0'; '0' 'ki' '0'; '0' '0' 'ki'});
        % Select all ellipse domains (all domains except the first one)
        all_domains = mphgetselection(model.geom('geom1').getRef().getDom);
        mat_inclusion.selection.set(all_domains.entities(2:end));

        
        % --- 5. Add Physics (Heat Transfer) ---
        phys = model.component('comp1').physics.create('ht', 'HeatTransfer', 'geom1');
        
        % BC for X-direction simulation
        temp_left = phys.create('temp1', 'Temperature', 1);
        temp_left.selection.named('geom1_sel_b_left');
        temp_left.set('T0', 'T_hot');
        
        temp_right = phys.create('temp2', 'Temperature', 1);
        temp_right.selection.named('geom1_sel_b_right');
        temp_right.set('T0', 'T_cold');
        
        % BC for Y-direction simulation
        temp_bottom = phys.create('temp3', 'Temperature', 1);
        temp_bottom.selection.named('geom1_sel_b_bottom');
        temp_bottom.set('T0', 'T_cold');
        
        temp_top = phys.create('temp4', 'Temperature', 1);
        temp_top.selection.named('geom1_sel_b_top');
        temp_top.set('T0', 'T_hot');
        
        % Define global parameters
        model.param.set('Lx', Lx, 'm');
        model.param.set('Ly', Ly, 'm');
        model.param.set('km', km, 'W/(m*K)');
        model.param.set('ki', ki, 'W/(m*K)');
        model.param.set('T_hot', T_hot, 'K');
        model.param.set('T_cold', T_cold, 'K');
        
        
        % --- 6. Meshing ---
        model.component('comp1').mesh.create('mesh1');
        model.component('comp1').mesh('mesh1').run();
        
        
        % --- 7. Setup and Run Studies ---
        % Study 1: X-direction gradient
        study1 = model.study.create('std1');
        study1.create('stat', 'Stationary');
        
        % Disable Y-dir BCs for Study 1
        model.physics('ht').feature('temp3').set('active', false);
        model.physics('ht').feature('temp4').set('active', false);
        
        sol1 = model.sol.create('sol1');
        sol1.study('std1');
        sol1.attach('std1');
        sol1.create('st1', 'StudyStep');
        sol1.create('v1', 'Variables');
        sol1.create('s1', 'Stationary');
        sol1.feature('s1').compile; % Pre-compile
        sol1.feature('s1').runAll;
        
        % Extract results from Study 1
        Qx1 = mphint(model, '-ht.ntflux.x', 'line', 'selection', model.geom('geom1').selection('sel_b_right').get());
        Qy1 = mphint(model, '-ht.ntflux.y', 'line', 'selection', [model.geom('geom1').selection('sel_b_top').get(), model.geom('geom1').selection('sel_b_bottom').get()]);

        
        % Study 2: Y-direction gradient
        study2 = model.study.create('std2');
        study2.create('stat', 'Stationary');
        
        % Re-enable Y-dir BCs and disable X-dir BCs
        model.physics('ht').feature('temp1').set('active', false);
        model.physics('ht').feature('temp2').set('active', false);
        model.physics('ht').feature('temp3').set('active', true);
        model.physics('ht').feature('temp4').set('active', true);
        
        sol2 = model.sol.create('sol2');
        sol2.study('std2');
        sol2.attach('std2');
        sol2.create('st1', 'StudyStep');
        sol2.create('v1', 'Variables');
        sol2.create('s1', 'Stationary');
        sol2.feature('s1').compile;
        sol2.feature('s1').runAll;
        
        % Extract results from Study 2
        Qy2 = mphint(model, '-ht.ntflux.y', 'line', 'selection', model.geom('geom1').selection('sel_b_top').get());
        Qx2 = mphint(model, '-ht.ntflux.x', 'line', 'selection', [model.geom('geom1').selection('sel_b_left').get(), model.geom('geom1').selection('sel_b_right').get()]);

        
        % --- 8. Calculate Effective Conductivity Matrix ---
        gradTx = -deltaT / Lx;
        gradTy = -deltaT / Ly;

        k_xx = -(Qx1 / Ly) / gradTx;
        k_yx = -(Qy1 / Lx) / gradTx; % Note the length normalization
        
        k_yy = -(Qy2 / Lx) / gradTy;
        k_xy = -(Qx2 / Ly) / gradTy; % Note the length normalization

        fprintf('  k_xx=%.4f, k_xy=%.4f, k_yx=%.4f, k_yy=%.4f\n\n', k_xx, k_xy, k_yx, k_yy);
        
        % --- 9. Store results ---
        results_data(i, :) = {sample_id, k_xx, k_xy, k_yx, k_yy, phi_actual};

    catch e
        fprintf('  ERROR processing file %s: %s\n', jsonFiles(i).name, e.message);
        results_data(i, :) = {sample_id, NaN, NaN, NaN, NaN, phi_actual};
    end
    
    % --- 10. Cleanup for next iteration ---
    model.modelNode.remove('mod1');
end

%% --- Save Results to CSV ---
results_table = cell2table(results_data, ...
    'VariableNames', {'sample_id', 'k_xx', 'k_xy', 'k_yx', 'k_yy', 'phi'});

writetable(results_table, outputCsvFile);

fprintf('All simulations finished. Results saved to %s\n', outputCsvFile);

% Disconnect from server
ModelUtil.disconnect;

