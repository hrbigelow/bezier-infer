import sys
import model as bmodel
import data as bdata


def main():
  try:
    train_data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    max_steps = int(sys.argv[3])
    checkpoint_path = sys.argv[4]
  except Exception as ex:
    print('\nUsage:\n\n'
        'train.py train_data.csv batch_size max_steps checkpoint_path\n\n'
        )
    raise SystemExit(0)

  model = bmodel.Model()
  model.cuda()

  train_loader = bdata.make_train_loader(train_data_path, batch_size)






