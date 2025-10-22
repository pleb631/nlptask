import torch

def predict_batch(model, inputs):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入,shape:[batch_size, sql_len]
    :return: 预测结果,shape:[batch_size]
    """
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
        # output.shape: [batch_size]
    batch_result = torch.softmax(output, dim=1)
    return batch_result


def predict(text, model, tokenizer, device,seq_len):
    # 1. 处理输入
    inputs =  tokenizer(text,padding='max_length', truncation=True,max_length=seq_len,return_tensors='pt')

    # 2.预测逻辑
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_result = predict_batch(model, inputs)

    return batch_result


