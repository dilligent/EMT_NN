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
jsonDir = 'generated_samples_bat\json_files'; 
% Output file for results
outputCsvFile = 'effective_conductivity_results.csv';

% Simulation Parameters
T_hot = 373.15;  % K (100 C)
T_cold = 273.15; % K (0 C)
deltaT = T_hot - T_cold;

%% --- Initialize COMSOL ---
% 检查是否需要启动连接
try
    % 尝试创建测试模型
    testModel = ModelUtil.create('Test');
    ModelUtil.remove('Test');
    fprintf('COMSOL LiveLink initialized successfully.\n');
catch
    % 如果失败，尝试启动连接
    fprintf('Initializing COMSOL connection...\n');
    mphstart;
end

%% --- Main Processing Loop ---

% Find all JSON files in the directory
jsonFiles = dir(fullfile(jsonDir, '*.json'));
numFiles = length(jsonFiles);
fprintf('Found %d JSON files to process.\n\n', numFiles);

% Initialize results table
results_data = cell(numFiles, 6); % sample_id, kxx, kxy, kyx, kyy, phi

for i = 1:numFiles
    modelTag = sprintf('Model_%d', i);
    model = ModelUtil.create(modelTag);
    model.component.create('comp1', true);

    jsonPath = fullfile(jsonFiles(i).folder, jsonFiles(i).name);
    fprintf('[%d/%d] Processing: %s\n', i, numFiles, jsonFiles(i).name);

    % 初始化，避免在解析前出错导致二次异常
    sample_id = string(jsonFiles(i).name);
    phi_actual = NaN;

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

        % --- 2. Build Geometry in COMSOL (保留夹杂域) ---
        geom = model.component('comp1').geom.create('geom1', 2);

        % 矩形基体
        r1 = geom.create('r1', 'Rectangle');
        r1.set('size', [Lx, Ly]);

        % 椭圆集合
        numEllipses = length(data.ellipses);
        ellipse_tags = cell(1, numEllipses);
        for j = 1:numEllipses
            ell = data.ellipses(j);
            tag = sprintf('el%d', j);
            ellipse_tags{j} = tag;
            elGeom = geom.create(tag, 'Ellipse');
            elGeom.set('pos', [ell.x, ell.y]);
            elGeom.set('semiaxes', [ell.a, ell.b]);
            elGeom.set('rot', ell.theta_deg);
        end
        % 椭圆并集
        uniE = geom.create('uniE', 'Union');
        uniE.selection('input').set(ellipse_tags);
        uniE.set('keep', true);

        % 基体域 = 矩形减椭圆
        dif1 = geom.create('dif1', 'Difference');
        dif1.selection('input').set('r1');
        dif1.selection('input2').set('uniE');
        dif1.set('keep', true);             % 保留选择
        dif1.set('createselection', 'on');  % 生成命名 selection

        % 夹杂域 = 矩形与椭圆交
        int1 = geom.create('int1', 'Intersection');
        int1.selection('input').set({'r1', 'uniE'});
        int1.set('keep', true);
        int1.set('createselection', 'on');

        % 最终成形（保留内边界，使其成为不同域）
        fin1 = geom.create('fin1', 'Union');
        fin1.selection('input').set({'dif1', 'int1'});
        fin1.set('intbnd', true); % keep interior boundaries

        geom.run;

        fprintf('  ✓ Geometry created: %d ellipses\n', numEllipses);

        % --- 3. Define Global Parameters (带单位表达式更稳妥) ---
        model.param.set('Lx', sprintf('%g[m]', Lx));
        model.param.set('Ly', sprintf('%g[m]', Ly));
        model.param.set('km', sprintf('%g[W/(m*K)]', km));
        model.param.set('ki', sprintf('%g[W/(m*K)]', ki));
        model.param.set('T_hot', sprintf('%g[K]', T_hot));
        model.param.set('T_cold', sprintf('%g[K]', T_cold));

        % --- 4. Boundary Selections（修正：使用 inside 避免选中垂直边界） ---
        % 关键修改：
        % 1. 使用 'inside' 条件，确保只有完全在框内的边界被选中。
        % 2. 扩大框的范围以完全包裹目标边界（例如左边界框跨越 x=0）。
        
        % Left Boundary (x=0)
        sel_b_left = geom.create('sel_b_left', 'BoxSelection');
        sel_b_left.set('entitydim', 1);
        sel_b_left.set('condition', 'inside'); % 仅选中完全在内部的实体
        sel_b_left.set('xmin', -0.1*Lx); sel_b_left.set('xmax', 0.1*Lx); % 跨越 x=0
        sel_b_left.set('ymin', -0.1*Ly); sel_b_left.set('ymax', 1.1*Ly); % 覆盖整个 y 高度

        % Right Boundary (x=Lx)
        sel_b_right = geom.create('sel_b_right', 'BoxSelection');
        sel_b_right.set('entitydim', 1);
        sel_b_right.set('condition', 'inside');
        sel_b_right.set('xmin', 0.9*Lx); sel_b_right.set('xmax', 1.1*Lx);
        sel_b_right.set('ymin', -0.1*Ly); sel_b_right.set('ymax', 1.1*Ly);

        % Bottom Boundary (y=0)
        sel_b_bottom = geom.create('sel_b_bottom', 'BoxSelection');
        sel_b_bottom.set('entitydim', 1);
        sel_b_bottom.set('condition', 'inside');
        sel_b_bottom.set('xmin', -0.1*Lx); sel_b_bottom.set('xmax', 1.1*Lx);
        sel_b_bottom.set('ymin', -0.1*Ly); sel_b_bottom.set('ymax', 0.1*Ly);

        % Top Boundary (y=Ly)
        sel_b_top = geom.create('sel_b_top', 'BoxSelection');
        sel_b_top.set('entitydim', 1);
        sel_b_top.set('condition', 'inside');
        sel_b_top.set('xmin', -0.1*Lx); sel_b_top.set('xmax', 1.1*Lx);
        sel_b_top.set('ymin', 0.9*Ly); sel_b_top.set('ymax', 1.1*Ly);

        geom.run;

        % --- 5. Materials（用几何特征 selection 绑定域） ---
        mat_matrix = model.component('comp1').material.create('mat1', 'Common');
        mat_matrix.label('Matrix');
        mat_matrix.propertyGroup('def').set('thermalconductivity', {'km'});
        % 绑定 dif1 的域
        mat_matrix.selection.named('geom1_dif1_dom');

        mat_inclusion = model.component('comp1').material.create('mat2', 'Common');
        mat_inclusion.label('Inclusion');
        mat_inclusion.propertyGroup('def').set('thermalconductivity', {'ki'});
        % 绑定 int1 的域
        mat_inclusion.selection.named('geom1_int1_dom');

        % --- 6. Add Physics (Heat Transfer) ---
        phys = model.component('comp1').physics.create('ht', 'HeatTransfer', 'geom1');
        
        % BC for X-direction simulation
        temp_left = phys.create('temp1', 'TemperatureBoundary', 1);
        temp_left.selection.named('geom1_sel_b_left');
        temp_left.set('T0', 'T_hot');
        
        temp_right = phys.create('temp2', 'TemperatureBoundary', 1);
        temp_right.selection.named('geom1_sel_b_right');
        temp_right.set('T0', 'T_cold');
        
        % BC for Y-direction simulation
        temp_bottom = phys.create('temp3', 'TemperatureBoundary', 1);
        temp_bottom.selection.named('geom1_sel_b_bottom');
        temp_bottom.set('T0', 'T_cold');
        
        temp_top = phys.create('temp4', 'TemperatureBoundary', 1);
        temp_top.selection.named('geom1_sel_b_top');
        temp_top.set('T0', 'T_hot');
        
        
        % --- 7. Meshing ---
        mesh = model.component('comp1').mesh.create('mesh1');
        mesh.autoMeshSize(4); % 4 = finer mesh, adjust as needed
        mesh.run();
        
        fprintf('  ✓ Mesh created\n');
        
        
        % --- 8. Setup and Run Studies ---
        % 创建积分耦合算子（只需创建一次）
        intop_right = model.component('comp1').cpl.create('intop_right', 'Integration');
        intop_right.selection.named('geom1_sel_b_right');
        
        intop_top = model.component('comp1').cpl.create('intop_top', 'Integration');
        intop_top.selection.named('geom1_sel_b_top');
        
        intop_bottom = model.component('comp1').cpl.create('intop_bottom', 'Integration');
        intop_bottom.selection.named('geom1_sel_b_bottom');
        
        intop_left = model.component('comp1').cpl.create('intop_left', 'Integration');
        intop_left.selection.named('geom1_sel_b_left');
        
        % Study 1: X-direction gradient (上下边界绝热)
        study1 = model.study.create('std1');
        study1.create('stat', 'Stationary');
        
        % Disable Y-dir BCs for Study 1 (上下边界绝热)
        phys.feature('temp3').active(false);
        phys.feature('temp4').active(false);
        
        % Run Study 1
        study1.run();
        
        fprintf('  ✓ Study 1 (X-direction) completed\n');
        
        % Extract results from Study 1
        % 右边界：热量流出（向外为正）
        Qx1 = mphglobal(model, 'intop_right(ht.ntflux)', 'dataset', 'dset1');
        % 上下边界：应该接近0（绝热）
        Qy1_top = mphglobal(model, 'intop_top(ht.ntflux)', 'dataset', 'dset1');
        Qy1_bottom = mphglobal(model, 'intop_bottom(ht.ntflux)', 'dataset', 'dset1');
        Qy1 = Qy1_top + Qy1_bottom;
        
        
        % Study 2: Y-direction gradient (左右边界绝热)
        study2 = model.study.create('std2');
        study2.create('stat', 'Stationary');
        
        % Re-enable Y-dir BCs and disable X-dir BCs
        phys.feature('temp1').active(false);
        phys.feature('temp2').active(false);
        phys.feature('temp3').active(true);
        phys.feature('temp4').active(true);
        
        % Run Study 2
        study2.run();
        
        fprintf('  ✓ Study 2 (Y-direction) completed\n');
        
        % Extract results from Study 2
        % 上边界：热量流出（向外为正）
        Qy2 = mphglobal(model, 'intop_top(ht.ntflux)', 'dataset', 'dset2');
        % 左右边界：应该接近0（绝热）
        Qx2_left = mphglobal(model, 'intop_left(ht.ntflux)', 'dataset', 'dset2');
        Qx2_right = mphglobal(model, 'intop_right(ht.ntflux)', 'dataset', 'dset2');
        Qx2 = Qx2_left + Qx2_right;
        

        % --- 9. Calculate Effective Conductivity Matrix ---
        % 修正后的公式
        
        % Study 1: X方向温度梯度
        % 左(x=0): T_hot, 右(x=Lx): T_cold
        % dT/dx = -deltaT/Lx
        % 根据 q = -K·∇T:
        % q_x = -k_xx * (-deltaT/Lx) = k_xx * deltaT/Lx
        % Q_x = q_x * Ly → k_xx = Q_x * Lx / (deltaT * Ly)
        k_xx = Qx1 * Lx / (deltaT * Ly);
        
        % q_y = -k_yx * (-deltaT/Lx) = k_yx * deltaT/Lx
        % Q_y = q_y * Lx → k_yx = Q_y / deltaT
        k_yx = Qy1 / deltaT;
        
        % Study 2: Y方向温度梯度
        % 下(y=0): T_cold, 上(y=Ly): T_hot
        % dT/dy = deltaT/Ly
        % q_y = -k_yy * (deltaT/Ly)
        % 从上边界流出的热量（向外为正）: Q_y = -q_y * Lx
        % 因为热量实际是向下流动，从上边界看是流入的
        % 所以: k_yy = -Qy2 * Ly / (deltaT * Lx)
        k_yy = -Qy2 * Ly / (deltaT * Lx);
        
        % q_x = -k_xy * (deltaT/Ly)
        % Q_x = -q_x * Ly → k_xy = -Qx2 / deltaT
        k_xy = -Qx2 / deltaT;

        fprintf('  k_xx=%.4f, k_xy=%.4f, k_yx=%.4f, k_yy=%.4f\n\n', k_xx, k_xy, k_yx, k_yy);
        
        % --- 10. Store results ---
        results_data(i, :) = {sample_id, k_xx, k_xy, k_yx, k_yy, phi_actual};

    catch e
        fprintf('  ✗ Error: %s\n', e.message);
        results_data(i, :) = {sample_id, NaN, NaN, NaN, NaN, phi_actual};
    end
    
    % Cleanup
    ModelUtil.remove(modelTag);
end

%% --- Save Results to CSV ---
results_table = cell2table(results_data, ...
    'VariableNames', {'sample_id', 'k_xx', 'k_xy', 'k_yx', 'k_yy', 'phi'});

writetable(results_table, outputCsvFile);

fprintf('\n✓ All done! Results saved to %s\n', outputCsvFile);

% Disconnect from server
ModelUtil.disconnect;

