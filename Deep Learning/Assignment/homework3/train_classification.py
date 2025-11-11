import argparse, torch, torch.nn as nn, torch.optim as optim
from homework.models import Classifier, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data

def train_one_epoch(model, loader, opt, scaler, device):
    model.train(); ce = nn.CrossEntropyLoss(); metric = AccuracyMetric()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x); loss = ce(logits, y)
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        with torch.no_grad(): metric.add(logits.argmax(1), y); total_loss += loss.item()*x.size(0); n += x.size(0)
    return total_loss/max(1,n), metric.compute()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); metric = AccuracyMetric()
    for x, y in loader:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        metric.add(model(x).argmax(1), y)
    return metric.compute()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="classification_data/train")
    ap.add_argument("--val_dir",   default="classification_data/val")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", choices=["adam","sgd","adamw"], default="adamw")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = "aug" if args.aug else "default"
    train_loader = load_data(args.train_dir, transform_pipeline=train_tf,
                             batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = load_data(args.val_dir,   transform_pipeline="default",
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Classifier(in_channels=3, num_classes=6).to(device)
    if args.optim == "adam":   opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optim == "sgd":  opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    else:                      opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type=="cuda")

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr = train_one_epoch(model, train_loader, opt, scaler, device)
        va = evaluate(model, val_loader, device)
        print(f"[{ep:03d}] loss={tr_loss:.4f} train_acc={tr['accuracy']:.4f} val_acc={va['accuracy']:.4f}")
        if va["accuracy"] > best:
            best = va["accuracy"]
            path = save_model(model)
            print(f"  saved best to {path} (val_acc={best:.4f})")

if __name__ == "__main__":
    main()
