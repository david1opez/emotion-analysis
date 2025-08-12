import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';

export class DockerLambdaAwsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const dockerFunc = new lambda.DockerImageFunction(this, 'EmotionAnalysisECRLF', {
      code: lambda.DockerImageCode.fromImageAsset('./image'),
      memorySize: 2000,
      timeout: cdk.Duration.seconds(90)
    });

    const functionUrl = dockerFunc.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.POST],
        allowedOrigins: ['*'],
        allowedHeaders: ['*'],
      }
    });

    new cdk.CfnOutput(this, 'functionUrl', {
      value: functionUrl.url,
    });
  }
}
