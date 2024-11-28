from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BanditTOMNetConfig:
    dim_state: int
    num_actions: int
    dim_action: int
    dim_encoder: int
    dim_decoder: int
    dim_latent: int
    encoder_layers: int
    decoder_layers: int
    action_to_emb: str = "embed"
    state_to_emb: str = None


class BanditToMNet(nn.Module):
    """See A.3.2 of ToMNet paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_state_to_emb(config)
        self.init_action_to_emb(config)
        self.char_net = CharNet(config)
        self.pred_net = PredictionNet(config)

    def forward(self, current_state, past_states, past_actions):
        # current_state: (bsz, num_agents, _)
        # current_state_emb: (bsz, num_agents, state_dim)
        # past_states: (bsz, seq_len, num_agents, _)
        # state_emb: (bsz, seq_len, num_agents, state_dim)
        # past_actions: (bsz, seq_len, num_agents, num_actions)
        # action_emb: (bsz, seq_len, num_agents, action_dim)
        current_state_emb = self.state_to_emb(current_state)
        state_emb = self.state_to_emb(past_states)
        action_emb = self.action_to_emb(past_actions)
        char_embed = self.char_net(state_emb, action_emb)
        action_logits = self.pred_net(char_embed, current_state_emb)
        return action_logits

    def init_state_to_emb(self, config):
        if config.state_to_emb is None:
            self.state_to_emb = lambda x: x
        else:
            raise NotImplementedError

    def init_action_to_emb(self, config):
        if config.action_to_emb is None:
            self.action_to_emb = lambda x: x
        elif config.action_to_emb == "embed":
            self.action_to_emb = nn.Embedding(
                num_embeddings=config.num_actions,
                embedding_dim=config.dim_action,
            )
        else:
            raise NotImplementedError


class CharNet(nn.Module):
    """character net parses an agent’s past trajectories from a set of POMDPs
    to form a character embedding
    """

    def __init__(self, config: BanditTOMNetConfig):
        super().__init__()
        self.config = config
        self.model = MLP(
            input_size=config.dim_state + config.dim_action,
            hidden_size=config.dim_encoder,
            output_size=config.dim_latent,
            num_hidden_layers=config.encoder_layers,
        )

    def forward(self, state_emb, action_emb):
        # state_emb: (bsz, num_agents, seq_len, state_dim)
        # action_emb: (bsz, num_agents, seq_len, action_dim)
        # char_embed: (bsz, num_agents, dim_lat)
        x = torch.cat([state_emb, action_emb], dim=-1)
        char_embed = self.model(x).mean(dim=-2)
        return char_embed


class PredictionNet(nn.Module):
    """prediction net takes the character embedding and the current stateervation
    of an agent as input and predicts the agent’s next action
    """

    def __init__(self, config: BanditTOMNetConfig):
        super().__init__()
        self.config = config
        self.model = MLP(
            input_size=config.dim_latent + config.dim_state,
            hidden_size=config.dim_decoder,
            output_size=config.num_actions,
            num_hidden_layers=config.decoder_layers,
        )

    def forward(self, char_embed, current_state_emb):
        # char_embed: (bsz, num_agents, dim_lat)
        # current_state: (bsz, num_agents, dim_state)
        x = torch.cat([char_embed, current_state_emb], dim=-1)
        action_logits = self.model(x)
        return action_logits


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.rms_norm(F.relu(x), (x.shape[-1],))
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.rms_norm(F.relu(x), (x.shape[-1],))
        x = self.output_layer(x)
        return x


@dataclass
class BanditFacultyConfig:
    dim_state: int
    num_actions: int
    num_teachers_total: int
    num_teachers_per_batch: int

    dim_observation: int
    observation_fn_layers: int
    observation_fn_dim: int
    policy_fn_layers: int
    policy_fn_dim: int

    seed_init: int
    seed_teachers: int


class BanditFaculty(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.observation_fns = nn.ModuleList(
            [self.init_observation_fn(config) for _ in range(config.num_teachers_total)]
        )
        self.policy_fn = self.init_policy_fn(config)
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(config.seed_init)
        self.init_weights()

        self.teachers_rng = torch.Generator()
        self.teachers_rng.manual_seed(config.seed_teachers)
        self.reset_teachers()

    def init_weights(self):
        # init weights with self.init_rng
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, generator=self.init_rng)
                nn.init.zeros_(m.bias)

    def init_observation_fn(self, config):
        return MLP(
            input_size=config.dim_state,
            hidden_size=config.observation_fn_dim,
            output_size=config.dim_observation,
            num_hidden_layers=config.observation_fn_layers,
        )

    def init_policy_fn(self, config):
        return MLP(
            input_size=config.dim_observation,
            hidden_size=config.policy_fn_dim,
            output_size=config.num_actions,
            num_hidden_layers=config.policy_fn_layers,
        )

    def forward(self, states: torch.Tensor, teacher_ids: list[int]):
        # states: (bsz, dim_state)
        # teachers: (ntpb)
        # observations: (bsz, ntpb, dim_observation)
        # action_logits: (bsz, ntpb, num_actions)
        # action_ids: (bsz, ntpb)

        # MBDO: how does this scale with multiple teachers? Parallelize?
        observations = torch.stack(
            [self.observation_fns[teacher_id](states) for teacher_id in teacher_ids],
            dim=0,
        )
        action_logits = self.policy_fn(observations)
        return action_logits

    def sample_actions(self, states: torch.Tensor, teacher_ids: list[int]):
        action_logits = self.forward(states, teacher_ids)
        # MBDO: alternative to argmax?
        action_ids = action_logits.argmax(dim=-1)
        return action_ids

    def reset_teachers(self):
        self.teachers = torch.randperm(
            self.config.num_teachers_total, generator=self.teachers_rng
        )

    def sample_teachers(self):
        if len(self.teachers) <= self.config.num_teachers_per_batch:
            temp_teachers = self.teachers.clone()
            self.reset_teachers()
            self.teachers = torch.cat([temp_teachers, self.teachers], dim=0)

        teachers = self.teachers[: self.config.num_teachers_per_batch]
        return teachers.tolist()


@dataclass
class TrainConfig:
    run_name: str

    # env setup
    num_agents: int = 8
    num_actions: int = 2
    dim_states: int = 16
    history_len: int = 4
    state_seed: int = 42

    # faculty setup
    dim_observations: int = 4
    faculty_n_layers: int = 1
    seed_init: int = 42
    seed_teachers: int = 42
    num_teachers_per_batch: int = 8

    # student setup
    student_n_layers: int = 1
    dim_actions: int = 8
    dim_student: int = 16

    # optimization setup
    bsz: int = 64
    num_train_steps: int = 10_000
    lr_warmup_steps: int = 1_000
    lr_peak: float = 1e-4
    lr_decay: float = 0.1
    adam_kwargs: dict = None

    # logging setup
    wandb_project: str = "ToMMM"
    wandb_entity: str = "abstraction"
    wandb_group: None | str = None
    wandb_tags: None | list[str] = None

    def init_faculty(self):
        config = BanditFacultyConfig(
            dim_state=self.dim_states,
            num_actions=self.num_actions,
            num_teachers_total=self.num_agents,
            num_teachers_per_batch=self.num_teachers_per_batch,
            dim_observation=self.dim_observations,
            observation_fn_layers=self.faculty_n_layers,
            observation_fn_dim=self.dim_states,
            policy_fn_layers=self.faculty_n_layers,
            policy_fn_dim=self.dim_states,
            seed_init=self.seed_init,
            seed_teachers=self.seed_teachers,
        )
        faculty = BanditFaculty(config)
        return faculty

    def init_student(self):
        config = BanditTOMNetConfig(
            dim_state=self.dim_states,
            num_actions=self.num_actions,
            dim_action=self.dim_actions,
            dim_encoder=self.dim_student,
            dim_decoder=self.dim_student,
            dim_latent=self.dim_student,
            encoder_layers=self.student_n_layers,
            decoder_layers=self.student_n_layers,
        )
        student = BanditToMNet(config)
        return student

    def init_optimizer(self, model):
        self.adam_kwargs = self.adam_kwargs or {}
        optimizer = torch.optim.AdamW(model.parameters(), **self.adam_kwargs)
        return optimizer

    def init_lr_scheduler(self, optimizer):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_train_steps,
            eta_min=self.lr_decay * self.lr_peak,
        )
        return lr_scheduler

    def sample_states(self):
        if getattr(self, "_state_rng", None) is None:
            self._state_rng = torch.Generator()
            self._state_rng.manual_seed(self.state_seed)

        current_states = torch.randn(
            self.bsz, self.dim_states, generator=self._state_rng
        )
        past_states = torch.randn(
            self.history_len, self.dim_states, generator=self._state_rng
        )

        return current_states, past_states

    def init_wandb(self):
        import wandb

        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.run_name,
            group=self.wandb_group,
            tags=self.wandb_tags,
            config=self.__dict__,
        )

    @property
    def ntpb(self):
        return self.num_teachers_per_batch


if __name__ == "__main__":
    import wandb 

    exp_name = "241127-tomnet"
    total_bsz = 1024
    dim_student = 16
    num_train_steps = 1024
    log_every = 1

    seeds = [42, 42**2, 42**3, 42**4, 42**5]
    num_agents = [1,2,4,8,16,32,64,128]
    model_scales = [1,2,4,8,16,32,64,128]
    # history_lens = [1,2,4,8,16]
    history_lens = [8]
    params = []
    for hl in history_lens:
        for ms in model_scales:
            for na in num_agents:
                for s in seeds:
                    params.append((na, ms, hl, s))

    print(f"Running {len(params)} experiments")
    for exp_idx, (na, ms, hl, s) in enumerate(params):
        run_name =f"{exp_name}-a={na}_m={ms}_h={hl}"
        ntpb = na
        bsz = total_bsz // na
        assert bsz * na == total_bsz
        cfg = TrainConfig(
            run_name=run_name, 
            wandb_group=exp_name,
            num_agents=na, 
            dim_student=dim_student * ms,
            history_len=hl,
            seed_init=s,
            seed_teachers=s,
            state_seed=s,
            num_teachers_per_batch=ntpb,
            bsz=bsz,
            num_train_steps=num_train_steps,
            lr_warmup_steps=num_train_steps,
            lr_decay=1.0
        )
        print("Initializing faculty...")
        faculty = cfg.init_faculty()
        print("Initializing student...")
        student = cfg.init_student()
        print("Initializing optimizer and scheduler...")
        optimizer = cfg.init_optimizer(student)
        lr_scheduler = cfg.init_lr_scheduler(optimizer)

        print(f"Training {run_name} ({exp_idx+1}/{len(params)})...")
        cfg.init_wandb()
        for i in range(cfg.num_train_steps):
            current_states, past_states = cfg.sample_states()
            with torch.inference_mode():
                teacher_ids = faculty.sample_teachers()
                actions = faculty.sample_actions(current_states, teacher_ids)
                past_actions = faculty.sample_actions(past_states, teacher_ids)

            # past_actions: (bsz, ntpb, seq)
            # past_states: (bsz, ntpb, seq, dim_state)
            # current_states: (bsz, ntpb, dim_state)
            past_actions = past_actions.clone().unsqueeze(0).repeat(cfg.bsz, 1, 1)
            past_states = past_states.unsqueeze(0).unsqueeze(0)
            past_states = past_states.repeat(cfg.bsz, cfg.ntpb, 1, 1)
            current_states = current_states.unsqueeze(1).repeat(1, cfg.ntpb, 1)

            action_logits = student.forward(current_states, past_states, past_actions)
            action_logits = action_logits.view(-1, cfg.num_actions)
            actions = actions.clone().view(-1)
            loss = F.cross_entropy(action_logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % log_every == 0:
                with torch.inference_mode():
                    acc = (action_logits.argmax(dim=-1) == actions).float().mean()
                wandb.log({"loss": loss.item(), "acc": acc.item()}, step=i)
                print(f"Step {i}: loss={loss.item()}, acc={acc.item()}", end="\r")

        print(f"Finished training {run_name} ({exp_idx+1}/{len(params)})")
        wandb.finish()
