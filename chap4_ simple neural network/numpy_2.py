import os
import urllib.request
import numpy as np
import tensorflow as tf

# TF1.x 图模式（避免 TF1 环境下各种 eager/tf.function 不兼容）
tf.compat.v1.disable_eager_execution()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

KERAS_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    print(f"[download] {url} -> {dst_path}")
    urllib.request.urlretrieve(url, dst_path)


def mnist_dataset(data_dir: str = "mnist_data"):
    """
    返回:
      (x_train, y_train), (x_test, y_test)
      x: float32, shape (N, 28, 28), range [0,1]
      y: int64,  shape (N,)
    """
    npz_path = os.path.join(data_dir, "mnist.npz")
    _download(KERAS_MNIST_URL, npz_path)

    with np.load(npz_path) as f:
        x_train = f["x_train"]  # uint8, (60000, 28, 28)
        y_train = f["y_train"]  # uint8, (60000,)
        x_test = f["x_test"]    # uint8, (10000, 28, 28)
        y_test = f["y_test"]    # uint8, (10000,)

    x_train = (x_train.astype(np.float32) / 255.0)
    x_test = (x_test.astype(np.float32) / 255.0)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


def build_model(x_flat, hidden=100):
    """两层全连接：784 -> hidden -> 10，返回 logits"""
    in_dim = 28 * 28
    out_dim = 10

    with tf.compat.v1.variable_scope("fnn", reuse=tf.compat.v1.AUTO_REUSE):
        W1 = tf.compat.v1.get_variable(
            "W1", shape=[in_dim, hidden],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.1)
        )
        b1 = tf.compat.v1.get_variable(
            "b1", shape=[hidden],
            initializer=tf.compat.v1.zeros_initializer()
        )
        W2 = tf.compat.v1.get_variable(
            "W2", shape=[hidden, out_dim],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.1)
        )
        b2 = tf.compat.v1.get_variable(
            "b2", shape=[out_dim],
            initializer=tf.compat.v1.zeros_initializer()
        )

        h1 = tf.nn.relu(tf.matmul(x_flat, W1) + b1)
        logits = tf.matmul(h1, W2) + b2
        return logits


def main():
    (train_x, train_y), (test_x, test_y) = mnist_dataset()

    batch_size = 256
    epochs = 10
    lr = 1e-3
    hidden = 100

    steps_per_epoch = int(np.ceil(train_x.shape[0] / batch_size))

    # --------- tf.data (TF1 没有 AUTOTUNE，直接用固定 prefetch buffer) ---------
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(60000)
        .repeat()               # 无限重复，按 steps_per_epoch 控制一个 epoch
        .batch(batch_size)
        .prefetch(1)
    )
    train_it = tf.compat.v1.data.make_one_shot_iterator(train_ds)
    xb_img, yb = train_it.get_next()  # xb_img: (B, 28, 28)

    xb = tf.reshape(xb_img, [-1, 28 * 28])
    yb = tf.cast(yb, tf.int64)

    # --------- train graph ---------
    logits_b = build_model(xb, hidden=hidden)
    loss_b = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yb, logits=logits_b)
    )
    pred_b = tf.argmax(logits_b, axis=1, output_type=tf.int64)
    acc_b = tf.reduce_mean(tf.cast(tf.equal(pred_b, yb), tf.float32))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_b)

    # --------- eval graph (placeholder 走同一套变量) ---------
    x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28], name="x_ph")
    y_ph = tf.compat.v1.placeholder(tf.int64, shape=[None], name="y_ph")
    x_ph_flat = tf.reshape(x_ph, [-1, 28 * 28])

    logits_eval = build_model(x_ph_flat, hidden=hidden)
    loss_eval = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits_eval)
    )
    pred_eval = tf.argmax(logits_eval, axis=1, output_type=tf.int64)
    acc_eval = tf.reduce_mean(tf.cast(tf.equal(pred_eval, y_ph), tf.float32))

    # --------- run ---------
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            # train
            tr_loss_sum = 0.0
            tr_acc_sum = 0.0

            for _ in range(steps_per_epoch):
                _, l, a = sess.run([train_op, loss_b, acc_b])
                tr_loss_sum += float(l)
                tr_acc_sum += float(a)

            tr_loss = tr_loss_sum / steps_per_epoch
            tr_acc = tr_acc_sum / steps_per_epoch

            # test (分 batch 评估，做加权平均更准)
            te_loss_sum = 0.0
            te_acc_sum = 0.0
            n_sum = 0

            for i in range(0, test_x.shape[0], batch_size):
                xb_np = test_x[i:i + batch_size]
                yb_np = test_y[i:i + batch_size]
                l, a = sess.run([loss_eval, acc_eval], feed_dict={x_ph: xb_np, y_ph: yb_np})
                bs = xb_np.shape[0]
                te_loss_sum += float(l) * bs
                te_acc_sum += float(a) * bs
                n_sum += bs

            te_loss = te_loss_sum / n_sum
            te_acc = te_acc_sum / n_sum

            print(
                f"epoch {epoch:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"test loss {te_loss:.4f} acc {te_acc:.4f}"
            )


if __name__ == "__main__":
    main()