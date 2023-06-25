import mlflow

def cal_sum(x,y):
    return x*y




if __name__=='__main__':
    # starting the server of ml flow
    with mlflow.start_run():
        x,y=75,10
        z=cal_sum(x,y)
        # tracking the experiment with mlflow
        mlflow.log_param('x',x)
        mlflow.log_param('y',y)
        mlflow.log_metric('z',z)

