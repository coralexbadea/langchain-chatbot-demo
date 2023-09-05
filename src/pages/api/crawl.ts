import { pipeline } from '@xenova/transformers';
import fs from "fs/promises"; // Import file system module
import { Document } from "langchain/document";
import { TokenTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { NextApiRequest, NextApiResponse } from "next";
import { supabaseAdminClient } from "utils/supabaseAdmin";
import { CustomEmbeddings } from "./customEmbeddings";
// import { summarizeLongDocument } from "./summarizer";

// The TextEncoder instance enc is created and its encode() method is called on the input string.
// The resulting Uint8Array is then sliced, and the TextDecoder instance decodes the sliced array in a single line of code.
const truncateStringByBytes = (str: string, bytes: number) => {
  const enc = new TextEncoder();
  return new TextDecoder("utf-8").decode(enc.encode(str).slice(0, bytes));
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    const generateEmbeddings = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2'
    );
    
    // Read the content of the 'train.txt' file
    const trainText = await fs.readFile("train.txt", "utf-8");

    const splitter = new TokenTextSplitter({
      encodingName: "gpt2",
      chunkSize: 300,
      chunkOverlap: 20,
    });

    // const pageContent = await summarizeLongDocument({ document: trainText });
    const pageContent = trainText
    const docs = splitter.splitDocuments([
      new Document({
        pageContent,
        metadata: {
          text: truncateStringByBytes(pageContent, 36000),
        },
      }),
    ]);

    const embeddings = new CustomEmbeddings(generateEmbeddings);
 
    const store = new SupabaseVectorStore(embeddings, {
      client: supabaseAdminClient,
      tableName: "documents",
    });

    await store.addDocuments(await docs);

    res.status(200).json({ message: "Done" });
  } catch (e) {
    console.log(e);
    res.status(500).json({ message: `Error ${JSON.stringify(e)}` });
  }
}
