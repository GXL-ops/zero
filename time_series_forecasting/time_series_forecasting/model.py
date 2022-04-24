import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


# 卷积块
class CovLayer(pl.LightningModule):
    def __init__(self, cov1_kernel=10):
        super().__init__()
        # 构建卷积层，将输入数据(batch, seq, feature), 使用一维卷积进行操作
        self.cov1_1 = torch.nn.Conv1d(1, 10, kernel_size=cov1_kernel)  # (N, 360, 1) => (N, 351, 10)

        self.cov1_2 = torch.nn.Conv1d(10, 128, kernel_size=cov1_kernel)  # (N, 351, 10) => (N, 342, 128)
        self.cov1_pool = torch.nn.MaxPool1d(kernel_size=3)  # (N, 114, 128)
        self.cov2_1 = torch.nn.Conv1d(128, 128, kernel_size=cov1_kernel)  # (N, 105, 128)
        self.cov2_2 = torch.nn.Conv1d(128, 128, kernel_size=cov1_kernel)  # (N, 96, 128)
        self.cov2_pool = torch.nn.MaxPool1d(kernel_size=3)  # (N, 32, 128)
        self.cov3_1 = torch.nn.Conv1d(128, 128, kernel_size=cov1_kernel)  # (N, 23, 128)
        self.cov3_2 = torch.nn.Conv1d(128, 128, kernel_size=cov1_kernel)  # (N, 14, 128)
        # 使用一个全局池化层将其进行池化操作，得到 (N, 14, 1)
        self.cov_adapt_pool = torch.nn.AdaptiveAvgPool1d(1)
        # 转化为 (N, 14) / 或者使用flatten() 之后使用一个全连接层，输出 (N, 1)
        self.cov_linear = torch.nn.Linear(14, 1)

        self.cov_flatten = nn.Flatten()
        self.cov_flatten_linear1 = nn.Linear(1792, 564)
        self.cov_flatten_linear2 = nn.Linear(564, 128)
        self.cov_flatten_linear3 = nn.Linear(128, 1)

    def forward(self, x):
        # (32, 1, 360) => (32, 128, 114)
        x = self.cov1_pool(F.relu(self.cov1_2(F.relu(self.cov1_1(x)))))
        print(x.shape)
        # (32, 128, 114) => (32, 128, 32)
        x = self.cov2_pool(F.relu(self.cov2_2(F.relu(self.cov2_1(x)))))
        # (32, 128, 32) => (32, 128, 14)
        x = F.relu(self.cov3_2(F.relu(self.cov3_1(x))))
        # (32, 128, 14) => (32, 128 * 128) => (32, 564)
        print(x.shape)
        x = self.cov_flatten_linear1(self.cov_flatten(x))
        # (32, 564) => (32, 128) => (32, 1) => (32, 1, 1)
        x = self.cov_flatten_linear3(self.cov_flatten_linear2(x)).unsqueeze(dim=2)

        return x


class TimeSeriesForcasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        features_len = 4,
        cov1_kernel=11,
        channels=512,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        # 构建卷积层，将输入数据(batch, seq, seq1, feature), 使用一维卷积进行操作
        self.cov_layers = []
        for i in range(features_len):
            self.cov_layers.append(CovLayer())

        # transformer模型
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        # transform 的 encoder操作不会对数据维度发生变化
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)

    def cov_src(self, src):
        """将传入特征维(batch, seq, seq2, feature)的序列经过卷积系列变化后转化为(batch, seq, feature)"""
        # src = (32, 200, 360, 4)
        return_src = []
        for i in range(src.shape[1]):
            # (32, 1, 360, 4)
            batch_seq_src = src[:, i]
            # (32, 360, 4)
            batch_seq_src = torch.squeeze(batch_seq_src, dim=1)
            seq_features = []
            for j in range(batch_seq_src.shape[2]):
                # (32, 360) => (32, 360, 1) => (32, 1, 360)
                x = batch_seq_src[:, :, j].unsqueeze(dim=2).permute(0, 2, 1)
                # (32, 1, 360) => (32, 1, 1)
                x = self.cov_layers[j](x)
                print(x.shape)
                seq_features.append(x)
            # (32, 1, 1) => (32, 1, 4)
            seq_features = torch.cat(seq_features, dim=2)
            return_src.append(seq_features)
        # (32, 1, 4) => (32, 200, 4)
        return_src = torch.cat(return_src, dim=1)
        print(return_src.shape)
        return return_src

    def encode_src(self, src):
        # (32, 200, 4) => (32, 200, 512) => (200, 32, 512)
        src_start = self.input_projection(src).permute(1, 0, 2)
        # 200, 32
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        # pos (200,) => (1, 200) => (32, 200)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # embedding将二维扩展到三维 (32, 200) => (32, 200, 512) => (200, 32, 512)
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        # 加法不是扩展（注意）(200, 32, 512) + (200, 32, 512) => (200, 32, 512)
        src = src_start + pos_encoder
        # (200, 32, 512) + (200, 32, 512) => (200, 32, 512)
        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):
        """

        :param trg: (32, 800, 1)
        :param memory: (200, 32, 512)
        :return:
        """
        # (32, 800, 1) => (32, 800, 512) => (800, 32, 512)
        trg_start = self.output_projection(trg).permute(1, 0, 2)
        # 800, 32
        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)
        # (800,) => (1, 800) => (32, 800)
        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        # (32, 800) => (32, 800, 512) => (800, 32, 512)
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        # (800, 32, 512) + (800, 32, 512) = (800, 32, 512)
        trg = pos_decoder + trg_start
        # 构建mask矩阵，时间序列逐步预测的mask (800, 800)
        trg_mask = gen_trg_mask(out_sequence_len, trg.device)
        # (800, 32, 512) + (800, 32, 512) => (800, 32, 512)
        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start
        # (32, 800, 512)
        out = out.permute(1, 0, 2)
        # (32, 800, 1)
        out = self.linear(out)

        return out

    def forward(self, x):
        # (32, 200, 360, 4), (32, 800, 1)
        src, trg = x
        # (32, 200, 360, 4) => (32, 200, 4)
        src = self.cov_src(src)
        # (32, 200, 4) => (200, 32, 512)
        src = self.encode_src(src)
        # (32, 800, 1), (200, 32, 512) => (32, 800, 1)
        out = self.decode_trg(trg=trg, memory=src)
        return out

    def training_step(self, batch, batch_idx):
        """

        :param batch: (32, 16, 9), (32, 16, 8), (32, 16, 1)
        :param batch_idx: 1
        :return:
        """
        # (32, 16, 9), (32, 16, 8), (32, 16, 1)
        src, trg_in, trg_out = batch
        # (32, 16, 1)
        y_hat = self((src, trg_in))
        # (512, )
        y_hat = y_hat.view(-1)
        # (32, 16, 1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)
        print("train_loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    n_classes = 100
    # (32, 200, 360, 4), (32, 800, 1)
    source = torch.rand(size=(32, 20, 360, 4))
    target_in = torch.rand(size=(32, 800, 1))
    target_out = torch.rand(size=(32, 800, 1))

    ts = TimeSeriesForcasting(n_encoder_inputs=4, n_decoder_inputs=1)
    print(ts)

    # # (32, 800 1)
    # pred = ts((source, target_in))
    #
    # print(pred.size())
    #
    # ts.training_step((source, target_in, target_out), batch_idx=1)
