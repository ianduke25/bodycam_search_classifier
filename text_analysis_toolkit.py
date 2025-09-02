import ast
import numpy as np
import os
import pandas as pd
import pickle  # Add this missing import
import re
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import spacy
import torch


class TextAnalysisToolkit:
    """
    A toolkit for text analysis, embedding generation, and bodycam search classification.
    """
    
    def __init__(self):
        """Initialize the toolkit."""
        self.related_sentences = [
            "Can you pop the thing so I can look at that sticker just to make sure that matches the paperwork?",
            "So you understand that I'm asking to search the vehicle.",
            "Is it okay if I look?",
            "Can I look at those bags in the back?",
            "Can you open up the back for me?",
            "Go to search the vehicle.",
            "And he's got marijuana.",
            "Did you search him?",
            "Use your body cam to take pictures.",
            "So let me finish searching the front",
            "I just wanted to finish getting this search.",
            "Ten six searched.",
            "I just wanted to finish getting this search.",
            "One set of plates, three fixed blade knives, one pair of bolt cutters.",
            "Alright, well, I'm going to take them and return to DMV because I'm assuming they're not registered to you.",
            "So what's on the property receipt?",
            "So apparently the window doesn't roll up all the way?",
            "I took my gloves off, man.",
            "I'm not going digging back in your center console again.",
            "All right, anything sharp, any needles, anything like that?",
            "You widen your stance a little bit.",
            "Anything in your boots?",
            "Any needles?",
            "Nothing tucked in your belt?",
            "We found a bottle of pills in the glove box.",
            "Step out for me.",
            "Anything on you?",
            "I'm going to search you real quick, okay?",
            "Hands on the hood.",
            "No needles?",
            "You can smell it from here.",
            "They stopped a guy because this rig just reeked of marijuana.",
            "Nothing sharp or anything in your pockets?",
            "You mind if we go in there and get whatever you're talking about?",
            "When you get out, I'll pat you down",
            "Were you drinking already, or was that already in here?",
            "What's in the backseat of my car?",
            "So we're going to take you back to my car, make sure you don't have anything on you you're not supposed to have, okay?",
            "Is there anything in the car that we need to be worried about?",
            "Any marijuana or anything?",
            "Guns, drugs, bazookas, bombs?",
            "What's in this pocket?",
            "I'm just going to detain you real quick, alright?",
            "Anything illegal in the car?",
            "To search it.",
            "And do you got anything that's going to poke me, make me bleed?",
            "Did you just recently use some math or anything like that?",
            "Did you just recently use some meth or anything like that?",
            "I  might have a knife in my left pocket",
            "In your left pocket?",
            "So you got anything illegal on your person?",
            "Something in the backpack",
            "So where's the knife located?",
            "Search the meth pipe, lesion, straw.",
            "Can you open your driver door for me so I can take a look at the door tag?",
            "Any marijuana or anything in here?",
            "Get your fucking hands out the window.",
            "Keep your hands up.",
            "Code for one in custody.",
            "You got search instrument to arrest me.",
            "Go glove up.",
            "If you want to get the front, I'll get the back.",
            "Watch out for sharps.",
            "Was that a weed pipe or a different pipe there at the floorboard?",
            "This here just weed.",
            "Yeah, the screwdriver and one blade.",
            "Under the center console.",
            "If you want to hop out, we'll do a quick patch, make sure you don't have anything you're not supposed to have, and we'll go from there.",
            "If you want to hop out, we'll do a quick pat, make sure you don't have anything you're not supposed to have, and we'll go from there.",
            "Anything illegal?",
            "I got pot in the car, but that's it.",
            "Heroin, needles?",
            "Any guns in the car?",
            "Put your hands on the car there",
            "Mind if I check the trunk for anything unusual?",
            "I need to verify the VIN number against your registration.",
            "Could you step to the side while we conduct the search?",
            "We're looking for any illegal substances or items.",
            "Please remain calm while we complete our inspection.",
            "Do you have any firearms in the vehicle?",
            "I noticed your taillight is out; I'll need to take a closer look.",
            "We received a report of suspicious activity in this area.",
            "I'm going to run your plates through the system.",
            "Have you been involved in any recent criminal activity?",
            "Please provide your driver's license and registration.",
            "We're conducting random security checks today.",
            "Do you consent to a search of your vehicle?",
            "I'm detecting the odor of illegal substances.",
            "We found a suspicious package under the seat.",
            "You're not carrying any stolen goods, are you?",
            "Have there been any alterations to your vehicle?",
            "We'll need to take a closer look at your documents.",
            "Is there a reason your vehicle smells like alcohol?",
            "You seem nervous; is there anything you'd like to tell me?",
            "Are these items yours or do they belong to someone else?",
            "We're going to need to detain this item for further investigation.",
            "Do you have anything in your pockets that I should know about?",
            "We're checking vehicles for safety compliance.",
            "Your vehicle matches the description of one reported stolen.",
            "I'm going to need backup to conduct a thorough search.",
            "Have you given anyone else permission to use your vehicle?",
            "There's been a report of illegal activity in this make and model.",
            "We need to verify the ownership of this vehicle.",
            "Your vehicle was seen leaving the scene of a crime.",
            "I'll need to document everything in your vehicle.",
            "Is there a legal reason you have this equipment?",
            "We're investigating a series of incidents in this neighborhood.",
            "Your cooperation is appreciated during this process.",
            "I'm going to check the vehicle's undercarriage.",
            "Are you aware it's illegal to transport these items?",
            "We'll need to test this substance for narcotics.",
            "Do you have any proof of purchase for these items?",
            "I'm going to need to see inside your glove compartment.",
            "You're required by law to comply with this search.",
            "Please explain why you have this amount of cash.",
            "We're conducting checks for national security reasons.",
            "Your vehicle has been identified in a recent investigation.",
            "I'll be recording this interaction for our records.",
            "Do you have any objection to me looking in the backseat?",
            "We need to ensure there are no contraband or weapons.",
            "Please step back while I inspect the exterior.",
            "Are these substances prescribed to you?",
            "I'll need to verify these serial numbers.",
            "Your license plate came back with several alerts.",
            "I'm required to inform you of your rights before the search.",
            "We've had reports of trafficking in this area.",
            "I'm checking for any modifications to your vehicle.",
            "This is a routine check for DUI enforcement.",
            "Your vehicle's description matches a recent alert.",
            "We're looking for a missing person; have you seen anyone suspicious?",
            "I'll need to take this for further examination.",
            "You have the right to refuse, but that may raise suspicion.",
            "I'll be checking for any hidden compartments.",
            "This area is known for drug smuggling.",
            "We need to clear your vehicle before you proceed.",
            "I'm going to run a check on these items.",
            "Please remain here while I call for a K-9 unit.",
            "Your cooperation can significantly speed up this process.",
            "We're conducting a safety inspection on all vehicles in this area.",
            "Please keep your hands where I can see them while I inspect the vehicle.",
            "I'm checking for any objects that might be considered a threat.",
            "This search is for our safety and yours.",
            "We're almost done here, just a few more areas to check.",
            "I appreciate your patience during this process.",
            "Everything seems in order, but I need to check one last thing.",
            "Your cooperation is making this much easier, thank you.",
            "I'm looking for anything that might be hidden out of plain sight.",
            "This is standard procedure, we do this for all traffic stops in this area.",
            "I'll need to look under the seats, can you please step out?",
            "Do you have a spare key? I need to open the trunk.",
            "It's policy to check all compartments for any illegal items.",
            "Have you had any issues with the vehicle's locking mechanisms?",
            "I'm going to use a flashlight to look into darker areas of the car.",
            "I noticed some irregularities with your vehicle's documentation.",
            "Your vehicle fits the description of one involved in recent incidents.",
            "We're almost finished; just need to verify a few more details.",
            "For documentation purposes, I need to take a few photos.",
            "Do you have any luggage or other large items in the vehicle?",
            "I need to confirm the serial numbers on some of your belongings.",
            "We're looking for specific items related to our current investigation.",
            "Your vehicle's registration number has been flagged for a routine check.",
            "I'm going to call in to confirm some details about your vehicle.",
            "Please provide the insurance information for your vehicle.",
            "Have you recently purchased anything of high value?",
            "I'm required to check the vehicle identification number directly.",
            "Your vehicle has been marked for a random safety inspection.",
            "I'll need to remove some of the items to get a better look.",
            "Can you explain why there's a discrepancy with your vehicle's records?",
            "We're conducting a thorough investigation, and your vehicle is part of it.",
            "There's been an alert for vehicles of this make and model.",
            "I'll need to cross-reference the VIN with the national database.",
            "Please detail the contents of any containers or packages in the car.",
            "We're ensuring that there are no hazards within the vehicle.",
            "I'll need to inspect any electronic devices found within the vehicle.",
            "Your vehicle's color and make match a description we've been given.",
            "For the next part of the inspection, I may need some additional tools.",
            "We're verifying the ownership of all high-value items in the vehicle.",
            "I'm checking for compliance with the latest safety regulations.",
            "Your patience is appreciated while we ensure everything is in order.",
            "The vehicle's condition suggests it might have been used for specific activities.",
            "We're collecting evidence as part of a larger investigation.",
            "I'll be looking for any modifications that might not comply with regulations.",
            "This inspection helps us ensure that all vehicles are safe and legal.",
            "Please confirm whether you've given anyone else access to your vehicle recently.",
            "We're nearly through; just a few more checks to complete.",
            "Your vehicle's type has been associated with certain risks, requiring a detailed check.",
            "I'll need to consult with my supervisor on a few details about this search.",
            "We're working to prevent and deter illegal activities in this area.",
            "This procedure is part of our commitment to public safety.",
            "I'm finalizing the report on this search; thank you for your cooperation.",
            'Fold in the cream cheese',
            "I'm hungry",
            "Back off! Im not joking around.",
            "Let's go buy a backpack"
        ]
        pass

    def encode_sentences_sbert(self, df, column_name):
        """
        Encode sentences using SBERT model.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing text data
        column_name (str): Name of the column containing text to encode
        
        Returns:
        pd.DataFrame: DataFrame with added 'embeddings' column
        """
        # Initialize SBERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize spaCy model for sentence segmentation
        nlp = spacy.load("en_core_web_sm")

        # Initialize a column for document-level embeddings
        df['embeddings'] = None
        df['sentences'] = None

        for index, row in df.iterrows():
            # Ensure the row value is a string before processing
            text = str(row[column_name])
            
            # Use spaCy for robust sentence tokenization
            sentences = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

            # Generate embeddings for all sentences in the document
            doc_embeddings = model.encode(sentences)

            # Store the list of embeddings for the document
            df.at[index, 'embeddings'] = [embedding.tolist() for embedding in doc_embeddings]
            df.at[index, 'sentences'] = sentences

        return df

    def collect_txt_files_data(self, directory_path):
        """
        Collect data from all .txt files in a directory.
        
        Parameters:
        directory_path (str): Path to the directory containing .txt files
        
        Returns:
        tuple: (filepaths, contents) - lists of file paths and their contents
        """
        filepaths = []
        contents = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    filepaths.append(filepath)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        contents.append(f.read())
        
        return filepaths, contents

    def save_data_to_csv(self, filepaths, contents, output_path):
        """
        Save filepath and content data to CSV.
        
        Parameters:
        filepaths (list): List of file paths
        contents (list): List of file contents
        output_path (str): Path where to save the CSV file
        """
        df = pd.DataFrame({
            'filepath': filepaths,
            'transcript': contents
        })
        
        df.to_csv(output_path, index=False)

    def clean_transcript(self, transcript):
        """
        Removes timestamps and unnecessary new lines from the transcript.
        
        Parameters:
        transcript (str): The transcript text.
        
        Returns:
        str: Cleaned transcript without timestamps and unnecessary new lines.
        """
        # Check if the input is a string
        if not isinstance(transcript, str):
            return transcript
        
        # Regular expression to match timestamps in the format 'number - number'
        timestamp_pattern = r'\[.*?\]'

        # Remove timestamps
        cleaned_transcript = re.sub(timestamp_pattern, '', transcript)
        
        # Remove unnecessary new lines and extra spaces
        cleaned_transcript = re.sub(r'\s*\n\s*', ' ', cleaned_transcript).strip()
        
        return cleaned_transcript

    def find_closest_sentences_sbert(self, df, transcript_column, embeddings_column, target_sentence, model):
        """
        Find sentences closest to a target sentence using SBERT embeddings.
        """
        target_phrase_column = "_".join(target_sentence.split()) + "_cosine_similarity"
        text_column = target_phrase_column + "_text"

        target_embedding = model.encode([target_sentence])[0]

        df[target_phrase_column] = np.nan
        df[text_column] = None

        for index, row in df.iterrows():
            sentence_embeddings = row[embeddings_column]
            sentences = row.get("sentences")

            if not sentence_embeddings or not sentences:
                continue

            # Cosine similarities
            similarities = [
                np.dot(target_embedding, np.array(emb)) / 
                (np.linalg.norm(target_embedding) * np.linalg.norm(emb))
                for emb in sentence_embeddings
            ]

            max_idx = int(np.argmax(similarities))
            df.at[index, target_phrase_column] = similarities[max_idx]
            df.at[index, text_column] = sentences[max_idx]

        return df

    def extract_filename(self, filepath):
        """
        Function to extract the string after the last '/' in a sequence.
        
        Parameters:
        filepath (str): Full file path
        
        Returns:
        str: Filename extracted from path
        """
        return filepath.split('/')[-1]

    def count_questions_and_sentences(self, row):
        """
        Count questions and sentences in a transcript.
        
        Parameters:
        row (pd.Series): DataFrame row containing transcript
        
        Returns:
        pd.Series: Series with num_questions and num_sentences counts
        """
        # Counting the number of question marks
        num_questions = row['transcript'].count('?')
        
        # Counting the number of sentences. Assuming sentences end with '.', '!', or '?'
        num_sentences = sum(row['transcript'].count(marker) for marker in ['.', '!', '?'])
        
        return pd.Series([num_questions, num_sentences], index=['num_questions', 'num_sentences'])

    def count_keywords_in_transcripts_case_insensitive(self, dataframe, keywords):
        """
        Count occurrences of keywords in transcripts (case-insensitive).
        
        Parameters:
        dataframe (pd.DataFrame): DataFrame containing transcript data
        keywords (list): List of keywords to count
        
        Returns:
        pd.DataFrame: DataFrame with added keyword count columns
        """
        # Adjusting the count_keywords function to correctly handle case-insensitive search
        def count_keywords_case_insensitive(text, keywords):
            text = text.lower()  # Convert text to lowercase
            return {keyword: text.count(keyword.lower()) for keyword in keywords}

        # Apply the counting function to the 'transcript' column and separate the counts into new columns
        for keyword in keywords:
            dataframe[keyword + '_count'] = dataframe['transcript'].apply(lambda x: count_keywords_case_insensitive(x, [keyword])[keyword])

        return dataframe

    def fit_logistic_and_find_best_score(self, X_train: pd.DataFrame, 
                                         y_train,
                                         alphas: np.array = [1e-3, 1e-3, 1e-2, 1e-1, 1, 5, 10],
                                         n_splits: int = 10,
                                         random_state: int = 13) -> dict:
        """
        Fit logistic regression with cross-validation to find best hyperparameters.
        
        Parameters:
        X_train (pd.DataFrame): Training features
        y_train: Training labels
        alphas (np.array): Regularization parameters to test
        n_splits (int): Number of CV folds
        random_state (int): Random state for reproducibility
        
        Returns:
        dict: Dictionary containing best parameters and performance metrics
        """
        best_alpha = alphas[0]
        best_recall = 0  # Focus on recall for optimization
        best_accuracy = 0
        best_precision = 0
        best_coefficients = None

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for alpha in alphas:
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Scale for each fold
                scaler = StandardScaler()
                X_train_fold_norm = scaler.fit_transform(X_train_fold)
                X_val_fold_norm = scaler.transform(X_val_fold)
                
                # Logistic Regression with L1 penalty
                # Define model (be sure to balance imbalanced data)
                model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear', random_state=random_state, class_weight='balanced', max_iter=1000)
                model.fit(X_train_fold_norm, y_train_fold)
                predictions = model.predict(X_val_fold_norm)
                
                # Calculate and store each metric
                fold_accuracies.append(accuracy_score(y_val_fold, predictions))
                fold_precisions.append(precision_score(y_val_fold, predictions, zero_division=0))
                fold_recalls.append(recall_score(y_val_fold, predictions, zero_division=1))

            # Compute mean of each metric across folds
            mean_accuracy = np.mean(fold_accuracies)
            mean_precision = np.mean(fold_precisions)
            mean_recall = np.mean(fold_recalls)

            # Update best scores and alpha if current mean recall is higher
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_accuracy = mean_accuracy
                best_precision = mean_precision
                best_alpha = alpha
                best_coefficients = model.coef_[0]

        # Combine feature names with coefficients
        feature_coeff_tuples = list(zip(X_train.columns, best_coefficients))

        # Return a dictionary of best scores, alpha, and coefficients
        return {
            'best_alpha': best_alpha,
            'best_accuracy': best_accuracy,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'feature_coefficients': feature_coeff_tuples
        }

    def classify_transcripts(self, transcript_directory, model_path):
        """
        Classify transcripts using a pre-trained model.
        
        Args:
            transcript_directory (str): Path to directory containing transcript txt files
            model_path (str): Path to the pickled classification model
            
        Returns:
            pd.DataFrame: DataFrame with filepaths and predictions
        """
        
        # Initialize models (no need to create new toolkit instance)
        nlp = spacy.load("en_core_web_sm")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the trained classifier
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        
        # Collect and process transcript data using self methods
        filepaths, contents = self.collect_txt_files_data(transcript_directory)
        
        df = pd.DataFrame({
            'filepath': filepaths,
            'transcript': contents
        })
        
        # Clean transcripts and generate embeddings using self methods
        df['transcript'] = df['transcript'].apply(self.clean_transcript)
        data = self.encode_sentences_sbert(df, 'transcript')
        
        # Clean transcription and embedding columns
        data['transcript'] = data['transcript'].apply(lambda x: str(x) if pd.notnull(x) else "")
        data['embeddings'] = data['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
        
        # Define reference sentences for similarity calculations
        all_sentences = self.related_sentences
        
        # Calculate similarity scores for each reference sentence using self methods
        for sentence in all_sentences:
            data = self.find_closest_sentences_sbert(data, 'transcript', 'embeddings', sentence, sentence_model)
        
        # Count questions and sentences using self methods
        data[['num_questions', 'num_sentences']] = data.apply(self.count_questions_and_sentences, axis=1)
        
        # Calculate aggregated embeddings
        data['mean_embedding'] = ''
        data['sum_embedding'] = ''
        data['mean_embeddings_final_5'] = ''
        data['mean_embeddings_first_5'] = ''
        
        for i in range(len(data)):
            embeddings = data['embeddings'].iloc[i]
            if len(embeddings) > 0:
                data.at[data.index[i], 'mean_embedding'] = np.mean(embeddings)
                data.at[data.index[i], 'sum_embedding'] = np.sum(embeddings)
                data.at[data.index[i], 'mean_embeddings_final_5'] = np.mean(embeddings[-5:])
                data.at[data.index[i], 'mean_embeddings_first_5'] = np.mean(embeddings[:5])
            else:
                data.at[data.index[i], 'mean_embedding'] = 0
                data.at[data.index[i], 'sum_embedding'] = 0
                data.at[data.index[i], 'mean_embeddings_final_5'] = 0
                data.at[data.index[i], 'mean_embeddings_first_5'] = 0
        
        # Add keyword counts using self methods
        keywords = ['confiscated', 'confiscate', 'search', 'marijuana', 'consent', 'weed', 'look', 'open', 'trunk']
        data = self.count_keywords_in_transcripts_case_insensitive(data, keywords)
        
        # Select only the features used by the trained model
        feature_columns = ['Can_you_pop_the_thing_so_I_can_look_at_that_sticker_just_to_make_sure_that_matches_the_paperwork?_cosine_similarity',
 "I'm_not_going_digging_back_in_your_center_console_again._cosine_similarity",
 "When_you_get_out,_I'll_pat_you_down_cosine_similarity",
 'Were_you_drinking_already,_or_was_that_already_in_here?_cosine_similarity',
 'Any_marijuana_or_anything?_cosine_similarity',
 'Keep_your_hands_up._cosine_similarity',
 'Your_cooperation_can_significantly_speed_up_this_process._cosine_similarity',
 'mean_embeddings_final_5',
 'marijuana_count',
 'trunk_count']
        
        # Set filepath as index and select feature columns
        data = data.set_index('filepath')
        feature_data = data[feature_columns]
        
        # Make predictions
        predictions = classifier.predict_proba(feature_data)
        
        # Create results DataFrame
        results = pd.DataFrame({
        'filepath': feature_data.index,
        'non_search_prob': predictions[:, 0], 
        'search_prob': predictions[:, 1],  
    }).reset_index(drop=True)
        
        return results