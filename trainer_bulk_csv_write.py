import os
import logging
from retry import retry
from neo4j import GraphDatabase

TRAINERS_CSV_PATH = os.getenv("TRAINERS_CSV_PATH")
QA_CSV_PATH = os.getenv("QA_CSV_PATH")
MEAL_PLAN_CSV_PATH = os.getenv("MEAL_PLAN_CSV_PATH")
MEAL_SUBMISSION_CSV_PATH = os.getenv("MEAL_SUBMISSION_CSV_PATH")
MOTIVATION_CSV_PATH = os.getenv("MOTIVATION_CSV_PATH")
TAG_CSV_PATH = os.getenv("TAG_CSV_PATH")
FEEDBACK_CSV_PATH = os.getenv("FEEDBACK_CSV_PATH")
MEAL_ITEM_CSV_PATH = os.getenv("MEAL_ITEM_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

NODES = ["Trainer", "QA", "MealPlan", "MealSubmission", "Motivation", "Tag", "Feedback"]

def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node}) REQUIRE n.id IS UNIQUE;"""
    _= tx.run(query, {})

@retry (tries=100, delay=5)
def load_trainer_data_from_csv() -> None:
    """Load trainer data from CSV data following a specific ontology into Neo4j.
    """

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    logger.info("Setting uniqueness constraints on nodes")

    with driver.session() as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    logger.info("Loading trainer nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{TRAINERS_CSV_PATH}" AS trainers
        MERGE (t:Trainer {{
            id: trainers.trainer_id,
            name: trainers.name,
            experience: toInteger(trainers.experience),
            tone: trainers.tone,
            vibe: trainers.vibe,
            tone_example: trainers.tone_example,
            certifications: split(trainers.certifications, ', '),
            specialties: split(trainers.specialties, ', '),
            sns_links: trainers.sns_links,
            location: trainers.location,
            avatar_url: trainers.avatar_url,
            subscription_price: trainers.subscription_price,
            available_hours: trainers.available_hours,
            bio: trainers.bio
        }})
        """
        _ = session.run(query, {})

    logger.info("Loading QA nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{QA_CSV_PATH}" AS qa
        MERGE (q:QA {{
            id: qa.qa_id,
            question: qa.question,
            answer: qa.answer,
            trainer_id: qa.trainer_id
        }})
        """
        _ = session.run(query, {})

    logger.info("Loading MealPlan nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{MEAL_PLAN_CSV_PATH}" AS meal_plan
        MERGE (mp:MealPlan {{
            id: meal_plan.meal_id,
            trainer_id: meal_plan.trainer_id,
            goal: meal_plan.goal,
            meal_type: meal_plan.meal_type,
            diet_type: meal_plan.diet_type,
            items: split(meal_plan.items, ', '),
            tags: split(meal_plan.tags, '|'),
            text: meal_plan.text,
            embedding: meal_plan.embedding,
            created_at: datetime(meal_plan.created_at),
            is_public: toBoolean(meal_plan.is_public)
        }})
        """
        _ = session.run(query, {})
    
    logger.info("Loading MealSubmission nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{MEAL_SUBMISSION_CSV_PATH}" AS meal_submission
        MERGE (ms:MealSubmission {{ 
            id: meal_submission.submission_id,
            user_id: meal_submission.user_id,
            image_url: meal_submission.image_url,
            meal_type: meal_submission.meal_type,
            nutrient_result: meal_submission.nutrient_result,
            feedback_text: meal_submission.feedback_text,
            analysis_source: meal_submission.analysis_source,
            detected_items: split(meal_submission.detected_items, ', '),
            verified: toBoolean(meal_submission.verified),
            meal_description: meal_submission.meal_description,
            submitted_at: datetime(meal_submission.submitted_at),
            trainer_id: meal_submission.trainer_id,
            flagged: toBoolean(meal_submission.flagged)
        }})
        """
        _ = session.run(query, {})

    logger.info("Loading MealItem nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{MEAL_ITEM_CSV_PATH}" AS meal_item
        MERGE (mi:MealItem {{
            id: meal_item.item_id,
            name: meal_item.name,
            base_food_name: meal_item.base_food_name,
            amount: meal_item.amount,
            calories: toFloat(meal_item.calories),
            protein: toFloat(meal_item.protein),
            carbs: toFloat(meal_item.carbs),
            fat: toFloat(meal_item.fat),
            sodium: toFloat(meal_item.sodium),
            tags: split(meal_item.tags, '|'),
            diet_type: meal_item.diet_type,
            is_common: meal_item.is_common,
            embedding: meal_item.embedding,
            created_at: datetime(meal_item.created_at)
        }})
        """
        _ = session.run(query, {})
    
    logger.info("Loading Motivation nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{MOTIVATION_CSV_PATH}" AS motivation
        MERGE (m:Motivation {{
            id: motivation.motivation_id,
            content: motivation.content,
            trainer_id: motivation.trainer_id
        }})
        """
        _ = session.run(query, {})
    
    logger.info("Loading Tag nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{TAG_CSV_PATH}" AS tag
        MERGE (t:Tag {{
            id: tag.tag_id,
            name: tag.name,
            trainer_id: tag.trainer_id      
        }})
        """
        _ = session.run(query, {})
    
    logger.info("Loading Feedback nodes")

    with driver.session() as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{FEEDBACK_CSV_PATH}" AS feedback
        MERGE (f:Feedback {{
            id: feedback.feedback_id,
            content: feedback.content,
            trainer_id: feedback.trainer_id,
            date: feedback.date,
            rating: toInteger(feedback.rating)
        }})
        """
        _ = session.run(query, {})  

