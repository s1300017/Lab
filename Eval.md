
1
AI Advances · Follow publication
Member-only story
How Do I Evaluate Chunking Strategies For RAGsDon’t guess; here’s how to systematically approach it.
9 min read · 6 days ago
Thuwarakesh Murallie Follow
Listen Share More
Image from Lummi.ai
I’ve researched RAGs extensively and know that chunking is critical to any RAG pipeline.Many people I’ve talked with trust that better models could improve RAGs. Some put too much trust in vector
databases. Even those who agreed that chunking is important didn’t think it could significantly improve the
system.Most of them argue that large context windows would replace the need for chunking strategies.But chunking techniques are here to stay. They are effective and a must for any RAG project.
6/3/25, 2:50 PM How Do I Evaluate Chunking Strategies For RAGs | by Thuwarakesh Murallie | May, 2025 | AI Advances
https://ai.gopubby.com/how-do-i-evaluate-chunking-strategies-for-rags-561cc5f9798b 1/18
However, a key question remains unanswered: How can I pick the best chunking strategy for a project?
One of These Will Be Your Favorite Chunking Technique For RAGs.
Document splitting and RAGs beyond the basics.
ai.gopubby.com
In the past, I’ve discussed several strategies: recursive character splitting, semantic chunking, and agentic
chunking, and even argued clustering as a fast and cheap alternative to agentic chunking.However, the bitter truth is that no strategy always works well. You can’t guess which would work based on
the project's nature.
The Most Valuable LLM Dev Skill is Easy to Learn, But Costly to Practice.
Here’s how not to waste your budget on evaluating models and systems.
medium.com
The only way is to try them all and find the best.So here’s my approach.
Start by sampling your documents.
If you’re working on a production app, you may have hundreds of gigabytes of data to process.But you can’t experiment with chunking strategies with all these data.
Cost is an apparent reason. Evaluations use LLM inferences. It may be an API call or a locally hosted LLM
inference, but both come with a price tag.
Exactly How Much vRAM (and Which GPU) Can Serve Your LLM?
What if you still want to run a massive model in your small GPU?
ai.gopubby.com
The other concern is the evaluation duration. Your project may be on a timeline (it often is), so you don’t
want to waste weeks evaluating chunking techniques.Want to move cheap & fast? Sample it.But make sure your sample accurately represents your population.That last one is easier said than done. I don’t want to discuss sampling techniques in a post dedicated to
chunking techniques, but given the nature of the project, stratified sampling may be a good choice.
6/3/25, 2:50 PM How Do I Evaluate Chunking Strategies For RAGs | by Thuwarakesh Murallie | May, 2025 | AI Advances
https://ai.gopubby.com/how-do-i-evaluate-chunking-strategies-for-rags-561cc5f9798b 2/18
For instance, if your organization has 100 client deliverable PPTs, 50 case study PDF documents, and 300
meeting transcripts, you can pick 10% in each as a sample. That way, you ensure every group has its
representation in the sample.Once you have the sample, you can move to the next step.
Create a test question set.
For evaluation, you need questions. The LLM will evaluate your answers to these questions.It would be best if a human expert manually created it. However, not every organization has that luxury.
Subject matter experts’ time is precious and often unavailable for R&D work.In such cases, you can get assistance from an LLM.LLMs can create questions from the sample documents. Though I keep saying augmentation in RAG
doesn’t need powerful LLMs, this step does. Pick a model with good reasoning abilities to create a high-
quality question set.My suggestions? If you’re on an Open AI stack, gpt-4o-mini is good enough. If not, try DeepSeek-R1-Distill-
Qwen-32B.The following code may help you get started. Make tweaks as needed for your specific requirements.
def create_test_questions(documents: List[Document], num_questions: int = 10) -> List[Dict[str, str]]        """Generate test questions from documents"""
        questions = []                # Sample text for question generation
        sample_texts = []        for doc in documents[:3]:  # Use first 3 docs
            words = doc.page_content.split()[:500]  # First 500 words
            sample_texts.append(" ".join(words))                combined_text = "\n\n".join(sample_texts)                prompt = f"""Based on the following text, generate {num_questions} diverse questions that can                Text:        {combined_text}
                Return only the questions, one per line, without numbering or additional text."""
                response = llm.invoke(prompt)        question_lines = [q.strip() for q in response.content.split('\n') if q.strip()]                for i, question in enumerate(question_lines[:num_questions]):            questions.append({                "question": question,                "question_id": f"q_{i+1}"
            })                return questions
6/3/25, 2:50 PM How Do I Evaluate Chunking Strategies For RAGs | by Thuwarakesh Murallie | May, 2025 | AI Advances
https://ai.gopubby.com/how-do-i-evaluate-chunking-strategies-for-rags-561cc5f9798b 3/18
