import { Embeddings } from "langchain/embeddings/base";


export class CustomEmbeddings extends Embeddings {
    private generateEmbeddings;
    constructor(generateEmbeddings:any){
        super({})
        this.generateEmbeddings = generateEmbeddings
    }

    async embedDocuments(documents: string[]): Promise<number[][]> {
      const embeddingPromises = documents.map(async (doc) => {
        return await this.generateEmbeddings(doc, {
          pooling: 'mean',
          normalize: true,
        });
      });
      const embeddings = await Promise.all(embeddingPromises);
    
      return embeddings;
    }

    async embedQuery(document: string): Promise<number[]> {
      const embedding = await this.generateEmbeddings(document, {
        pooling: 'mean',
        normalize: true,
      });
    
      // embedding is already an array, so no further transformation is needed
      return embedding;
    }
    
}
