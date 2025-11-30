import {
  loadImageModel,
  PreTrainedImageModels,
  getImageFeatures,
  loadImageClassifierModel,
  topClassifyResult,
  tf,
} from 'tensorflow-helpers'
import { readdirSync } from 'fs'
import { join } from 'path'

async function main() {
  // Load pre-trained base model
  let baseModel = await loadImageModel({
    spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })

  // Create classifier for image classification
  let classifier = await loadImageClassifierModel({
    baseModel,
    modelDir: 'saved_model/gender_classifier_model',
    hiddenLayers: [128],
    datasetDir: 'datasets/gender',
    classNames: ['MaineCoon', 'others','PersianCat','SiameseCat'], // auto scan from datasetDir
  })

  let correct = 0
  let total = 0


  let dataset_dir = 'test-datasets/gender'
  for(let label of readdirSync(dataset_dir)) {
    // console.log('test:', label)
    let filenames = readdirSync(join(dataset_dir, label))
    for(let filename of filenames) {
      let file = join(dataset_dir, label, filename)
      let result = await classifier.classifyImageFile(file)
      let topClass = topClassifyResult(result)
      // console.log('result:',topClass)
      if(topClass.label === label) {
        correct ++
    } 
      total ++
      let percentage = ((correct / total) * 100).toFixed(2) + '%'
      process.stdout.write(
        `\r correct: ${correct}, total: ${total}, accuracy: ${percentage} `,
      )
  }
  console.log()
  }
}
main().catch(e => console.error(e))