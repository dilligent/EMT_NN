"""
Deep Sets模型用于预测椭圆掺杂介质的等效热导率矩阵

输入: 矩形内掺杂的椭圆集合,每个椭圆由5个参数描述 (x, y, a, b, theta_deg)
输出: 2x2等效热导率矩阵 (k_xx, k_xy, k_yx, k_yy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetsEncoder(nn.Module):
    """
    Deep Sets编码器:将每个椭圆映射到高维特征空间
    """
    def __init__(self, input_dim=5, hidden_dims=[64, 128, 256]):
        super(DeepSetsEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_ellipses, input_dim) 每个椭圆的特征
        Returns:
            encoded: (batch_size, num_ellipses, output_dim) 编码后的特征
        """
        batch_size, num_ellipses, input_dim = x.shape
        # 重塑为 (batch_size * num_ellipses, input_dim)
        x_flat = x.view(-1, input_dim)
        # 编码
        encoded_flat = self.encoder(x_flat)
        # 重塑回 (batch_size, num_ellipses, output_dim)
        encoded = encoded_flat.view(batch_size, num_ellipses, -1)
        return encoded


class DeepSetsAggregator(nn.Module):
    """
    Deep Sets聚合器:将集合聚合为置换不变的表示
    支持多种聚合方式
    """
    def __init__(self, aggregation='mean'):
        super(DeepSetsAggregator, self).__init__()
        self.aggregation = aggregation
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_ellipses, feature_dim)
        Returns:
            aggregated: (batch_size, feature_dim)
        """
        if self.aggregation == 'mean':
            return torch.mean(x, dim=1)
        elif self.aggregation == 'sum':
            return torch.sum(x, dim=1)
        elif self.aggregation == 'max':
            return torch.max(x, dim=1)[0]
        elif self.aggregation == 'mean_max':
            # 同时使用mean和max
            mean_features = torch.mean(x, dim=1)
            max_features = torch.max(x, dim=1)[0]
            return torch.cat([mean_features, max_features], dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")


class DeepSetsDecoder(nn.Module):
    """
    Deep Sets解码器:将聚合后的特征映射到输出
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=4):
        super(DeepSetsDecoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) 聚合后的特征
        Returns:
            output: (batch_size, output_dim) 预测的热导率矩阵元素
        """
        return self.decoder(x)


class DeepSetsModel(nn.Module):
    """
    完整的Deep Sets模型
    """
    def __init__(self, 
                 ellipse_feature_dim=5,
                 encoder_hidden_dims=[64, 128, 256],
                 decoder_hidden_dims=[256, 128, 64],
                 aggregation='mean_max',
                 output_dim=4,
                 include_global_features=True):
        """
        Args:
            ellipse_feature_dim: 每个椭圆的特征维度 (x, y, a, b, theta_deg)
            encoder_hidden_dims: 编码器的隐藏层维度
            decoder_hidden_dims: 解码器的隐藏层维度
            aggregation: 聚合方式 ('mean', 'sum', 'max', 'mean_max')
            output_dim: 输出维度 (k_xx, k_xy, k_yx, k_yy)
            include_global_features: 是否包含全局特征 (phi, Lx, Ly, km, ki)
        """
        super(DeepSetsModel, self).__init__()
        
        self.include_global_features = include_global_features
        
        # 编码器
        self.encoder = DeepSetsEncoder(ellipse_feature_dim, encoder_hidden_dims)
        
        # 聚合器
        self.aggregator = DeepSetsAggregator(aggregation)
        
        # 计算解码器的输入维度
        aggregated_dim = encoder_hidden_dims[-1]
        if aggregation == 'mean_max':
            aggregated_dim *= 2
        
        # 如果包含全局特征,增加输入维度
        decoder_input_dim = aggregated_dim
        if include_global_features:
            # 全局特征: phi, Lx, Ly, km, ki = 5维
            decoder_input_dim += 5
        
        # 解码器
        self.decoder = DeepSetsDecoder(decoder_input_dim, decoder_hidden_dims, output_dim)
    
    def forward(self, ellipse_features, global_features=None):
        """
        Args:
            ellipse_features: (batch_size, num_ellipses, 5) 椭圆特征
            global_features: (batch_size, 5) 全局特征 [phi, Lx, Ly, km, ki]
        Returns:
            output: (batch_size, 4) 预测的热导率矩阵 [k_xx, k_xy, k_yx, k_yy]
        """
        # 编码每个椭圆
        encoded = self.encoder(ellipse_features)
        
        # 聚合
        aggregated = self.aggregator(encoded)
        
        # 如果包含全局特征,进行拼接
        if self.include_global_features and global_features is not None:
            aggregated = torch.cat([aggregated, global_features], dim=1)
        
        # 解码
        output = self.decoder(aggregated)
        
        return output


class DeepSetsWithAttention(nn.Module):
    """
    带注意力机制的Deep Sets模型
    """
    def __init__(self,
                 ellipse_feature_dim=5,
                 encoder_hidden_dims=[64, 128, 256],
                 decoder_hidden_dims=[256, 128, 64],
                 output_dim=4,
                 include_global_features=True,
                 attention_heads=4):
        super(DeepSetsWithAttention, self).__init__()
        
        self.include_global_features = include_global_features
        
        # 编码器
        self.encoder = DeepSetsEncoder(ellipse_feature_dim, encoder_hidden_dims)
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=encoder_hidden_dims[-1],
            num_heads=attention_heads,
            batch_first=True
        )
        
        # 计算解码器的输入维度
        decoder_input_dim = encoder_hidden_dims[-1]
        if include_global_features:
            decoder_input_dim += 5
        
        # 解码器
        self.decoder = DeepSetsDecoder(decoder_input_dim, decoder_hidden_dims, output_dim)
    
    def forward(self, ellipse_features, global_features=None):
        """
        Args:
            ellipse_features: (batch_size, num_ellipses, 5)
            global_features: (batch_size, 5)
        Returns:
            output: (batch_size, 4)
        """
        # 编码
        encoded = self.encoder(ellipse_features)
        
        # 自注意力
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # 平均池化
        aggregated = torch.mean(attended, dim=1)
        
        # 拼接全局特征
        if self.include_global_features and global_features is not None:
            aggregated = torch.cat([aggregated, global_features], dim=1)
        
        # 解码
        output = self.decoder(aggregated)
        
        return output


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    num_ellipses = 137
    
    # 创建随机数据
    ellipse_features = torch.randn(batch_size, num_ellipses, 5)
    global_features = torch.randn(batch_size, 5)
    
    # 测试基础Deep Sets模型
    print("Testing basic Deep Sets model...")
    model = DeepSetsModel(
        ellipse_feature_dim=5,
        encoder_hidden_dims=[64, 128, 256],
        decoder_hidden_dims=[256, 128, 64],
        aggregation='mean_max',
        include_global_features=True
    )
    
    output = model(ellipse_features, global_features)
    print(f"Input shape: {ellipse_features.shape}")
    print(f"Global features shape: {global_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 测试注意力模型
    print("\nTesting Deep Sets with Attention...")
    attention_model = DeepSetsWithAttention(
        ellipse_feature_dim=5,
        encoder_hidden_dims=[64, 128, 256],
        decoder_hidden_dims=[256, 128, 64],
        include_global_features=True,
        attention_heads=4
    )
    
    output_att = attention_model(ellipse_features, global_features)
    print(f"Output shape: {output_att.shape}")
    print(f"Total parameters: {sum(p.numel() for p in attention_model.parameters())}")
