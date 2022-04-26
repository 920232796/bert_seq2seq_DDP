
import torch
from tqdm import tqdm
import os
import argparse
from .launch import launch_dist

class Trainer:

    def __init__(self, epoches,
                 env_type="pytorch",
                 val_every_step=100,
                 batch_size=1,
                 device="cpu",
                 model_save_dir="./",
                 seed=1,

                 ## ditributed param
                 master_ip='localhost',
                 master_port=17750,
                 num_nodes=1,
                 num_gpus=1,
                 training_script="train.py",
                 ):
        pass
        self.seed = seed
        self.env_type = env_type
        self.epochs = epoches
        self.device = device
        self.val_every_step = val_every_step
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.best_metric = 0.0
        self.batch_size = batch_size
        self.not_call_launch = True

        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            print("gpu数量不符")
            os._exit(0)

        if env_type == "DDP":
            self.get_dist_args()
            if not self.not_call_launch:
                launch_dist(num_nodes=num_nodes,
                            gpus_per_node=num_gpus,
                            master_addr=master_ip,
                            master_port=master_port,
                            training_script=training_script)

                os._exit(1)
            self.initialize_distributed()

        elif env_type == "pytorch":
            self.local_rank = 0
        else :
            print("不支持的env type")
            os._exit(0)

    def print_rank_0(self, message):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(message, flush=True)
        else:
            print(message, flush=True)

    def initialize_distributed(self):
        """Initialize torch.distributed."""
        if self.env_type == 'pytorch':
            self.print_rank_0('No need to initialize')
            return
        if self.env_type == 'DDP':
            torch.backends.cudnn.enabled = False
            if self.local_rank is not None:
                device = self.local_rank
            torch.cuda.set_device(device)
            # Call the init process
            init_method = 'tcp://'
            self.master_ip = os.getenv('MASTER_ADDR', 'localhost')
            self.master_port = os.getenv('MASTER_PORT', '6000')
            init_method += self.master_ip + ':' + self.master_port
            print(init_method, self.rank, device, self.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                world_size=self.world_size, rank=self.rank,
                init_method=init_method)

    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default = 0, help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="not call launch!")
        ds_args = parser.parse_args()
        self.rank = int(os.environ.get('RANK',0))
        self.local_rank = ds_args.local_rank
        self.not_call_launch = ds_args.not_call_launch

        self.world_size = int(os.environ.get('WORLD_SIZE',1))
        self.master_addr = os.environ.get('MASTER_ADDR','127.0.0.1')
        self.master_port = os.environ.get('MASTER_PORT','17500')

    def get_dataloader(self, dataset, collate_fn, shuffle=False):
        if dataset is None :
            return None
        if self.env_type == 'pytorch':
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=shuffle)
        else:
            num_replicas = self.world_size
            rank = self.rank
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=num_replicas,
                                                                      rank=rank,
                                                                      shuffle=False)
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               sampler=sampler,
                                               num_workers=1,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=collate_fn)

    def train(self,model, optimizer, train_dataset,
              val_dataset=None, compute_metric_func=None,
              validation_func=None, collate_fn=None,
              ):
        self.model = model
        self.validation_func = validation_func
        self.compute_metric = compute_metric_func
        self.optimizer = optimizer
        if self.env_type=='pytorch':
            self.model.to(self.device)
            train_shuffle = True

        else:
            self.model.cuda(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)
            train_shuffle = False

        self.train_dataloader = self.get_dataloader(train_dataset, collate_fn=collate_fn, shuffle=train_shuffle)
        self.val_dataloader = self.get_dataloader(val_dataset, collate_fn=collate_fn, shuffle=False)

        for epoch in range(self.epochs):
            self.epoch = epoch
            if self.env_type == "DDP":
                self.train_dataloader.sampler.set_epoch(self.seed + epoch)
            self.train_epoch()

    def train_epoch(self):
        report_loss = 0.0
        self.step = 0
        self.model.train()
        for data in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.step += 1
            if self.env_type == "pytorch":
                data = {x: data[x].to(torch.device(self.device)) for x in data if data[x] is not None}
            else :
                # multi gpu
                data = {x: data[x].to(torch.device("cuda", self.local_rank)) for x in data if data[x] is not None}

            if self.step % self.val_every_step == 0:
                if self.local_rank == 0:
                    print(f"loss is {report_loss}")
                with torch.no_grad():
                    self.model.eval()
                    if self.validation_func is not None:
                        self.validation_func()
                        metric = 0.0
                        val_loss = 0.0
                    else :
                        metric, val_loss = self.validate()

                    if self.local_rank == 0:
                        self.after_val(metric, val_loss)
                self.model.train()
                report_loss = 0.0

            loss_v = self.train_step(**data)
            report_loss += loss_v

    def train_step(self, **model_in):
        pass

        model_out = self.model(**model_in)
        loss = model_out["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self):

        if self.val_dataloader is not None:
            metric = 0
            total_lm_loss = 0.
            total_number = 0
            for val_data in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
                if self.env_type == "pytorch":
                    data = {x: val_data[x].to(torch.device(self.device)) for x in val_data if val_data[x] is not None}
                else :
                    # multi gpu
                    data = {x: val_data[x].to(torch.device("cuda", self.local_rank)) for x in val_data if val_data[x] is not None}

                model_out = self.model(**data)
                logits = model_out["logits"]
                loss = model_out["loss"].data.detach().float().item()
                total_lm_loss += loss
                if self.compute_metric is None:
                    metric += 0
                else :
                    metric += self.compute_metric(logits, labels=data["labels"])
                    total_number += 1

            self.model.train()
            if torch.cuda.is_available():
                loss_data = torch.cuda.FloatTensor(
                    [total_lm_loss, total_number] + [metric])
            else:
                loss_data = torch.FloatTensor(
                    [total_lm_loss, total_number] + [metric])

            if self.env_type == 'DDP':
                torch.distributed.all_reduce(loss_data)

            val_loss = loss_data[0] / loss_data[1]
            metric = loss_data[2] / loss_data[1]

        else:
            metric = 0
            val_loss = 0.0

        return metric, val_loss

    def after_val(self, metric, val_loss):
        if self.local_rank == 0:
            print("\n")
            print(f"metric is {metric}")
            print(f"validation loss is {val_loss}")
            with open(os.path.join(self.model_save_dir, "result.txt"), "a+") as f:
                f.write(f"epoch: {self.epoch}, step: {self.step}, metric is {metric} validation loss is {val_loss} \n")
            torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, "final_model.bin"))
            print(f"model is saved {os.path.join(self.model_save_dir, 'final_model.bin')}")
            if metric > self.best_metric:
                self.best_metric = metric
                torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, "best_model.bin"))
                print(f"best model is saved {os.path.join(self.model_save_dir, 'best_model.bin')}")
