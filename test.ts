import {
  loadImageModel,
  PreTrainedImageModels,
  getImageFeatures,
  loadImageClassifierModel,
  topClassifyResult,
  tf,
} from 'tensorflow-helpers'

async function main() {
  // Load pre-trained base model
  let baseModel = await loadImageModel({
    spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })
  console.log('embedding features:', baseModel.spec.features)
  // [print] embedding features: 1280

  // Create classifier for image classification
  let classifier = await loadImageClassifierModel({
    baseModel,
    modelDir: 'saved_model/gender_classifier_model',
    hiddenLayers: [128],
    datasetDir: 'datasets/gender',
    classNames: ['MaineCoon', 'others','PersianCat','SiameseCat'], // auto scan from datasetDir
  })

  // auto load training dataset
  let history = await classifier.train({
    epochs: 5,
    batchSize: 32,
   })

  // persist the parameters across restart
  await classifier.save()

  // auto load image from filesystem, resize and crop
  let classes = await classifier.classifyImageFile('test.jpg')
  console.log('classes:', classes)
  let topClass = topClassifyResult(classes)

  console.log('result:', topClass)
  // [print] result: { label: 'anime', confidence: 0.7991582155227661 }
}
main().catch(e => console.error(e))