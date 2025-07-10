import random
from typing import List, Dict

class PracticePrompts:
    def __init__(self):
        self.prompts = {
            'interview': [
                "Tell me about yourself and your background.",
                "What is your biggest strength and how has it helped you?",
                "Describe a challenging situation you faced and how you overcame it.",
                "What is your biggest weakness and how are you working to improve it?",
                "Where do you see yourself in 5 years?",
                "Why are you interested in this position/company?",
                "Tell me about a time you had to work with a difficult team member.",
                "What motivates you to do your best work?",
                "Describe a project you're particularly proud of.",
                "How do you handle stress and pressure?",
                "What questions do you have for us?"
            ],
            'academic': [
                "Explain a complex concept from your field to a non-expert.",
                "Describe your research interests and methodology.",
                "What is the most significant challenge facing your field today?",
                "Present your thesis/project in 2 minutes.",
                "How would you teach this subject to someone with no background?",
                "What are the ethical implications of your research?",
                "Describe a time you had to defend your academic work.",
                "What inspired you to pursue this field of study?",
                "How do you stay current with developments in your area?",
                "What would you do differently if you started your research again?"
            ],
            'presentation': [
                "Introduce yourself to a new team or audience.",
                "Present a solution to a common workplace problem.",
                "Explain why people should care about your favorite hobby.",
                "Give a product pitch for something you use daily.",
                "Describe how to do something you're good at.",
                "Present the pros and cons of a current event or trend.",
                "Explain a process or system you're familiar with.",
                "Give a toast at a wedding or celebration.",
                "Present quarterly results to stakeholders.",
                "Explain why your idea deserves funding or support."
            ],
            'conversational': [
                "What book/movie/show has influenced you most and why?",
                "Describe your ideal weekend or vacation.",
                "What skill would you most like to develop and why?",
                "Tell me about a person who has been a mentor to you.",
                "What's the best advice you've ever received?",
                "Describe a tradition that's important to you.",
                "What would you do if you won the lottery?",
                "Tell me about a time you stepped outside your comfort zone.",
                "What's something you've learned recently that surprised you?",
                "Describe your hometown to someone who's never been there."
            ],
            'ielts_speaking': [
                "Describe a memorable journey you have taken.",
                "Talk about a skill you would like to learn.",
                "Describe a person who has influenced your life.",
                "Explain a traditional celebration in your country.",
                "Describe a place you would like to visit.",
                "Talk about a book that you enjoyed reading.",
                "Describe a time when you helped someone.",
                "Explain how technology has changed education.",
                "Describe your ideal job.",
                "Talk about environmental problems in your area."
            ],
            'cat_preparation': [
                "What are your career goals and how does an MBA fit into them?",
                "Describe a leadership experience you've had.",
                "What is your opinion on corporate social responsibility?",
                "How would you handle a situation where you disagree with your manager?",
                "Explain a current business trend and its implications.",
                "What makes a good team player?",
                "Describe a time you had to make a difficult decision.",
                "What are the challenges facing your industry?",
                "How do you define success?",
                "What would you bring to our MBA program?"
            ],
            'impromptu': [
                "If you could have dinner with any three people, who would they be?",
                "What superpower would you choose and why?",
                "Describe the perfect day.",
                "What would you do if you were invisible for a day?",
                "If you could change one thing about the world, what would it be?",
                "What's the most important lesson you've learned from failure?",
                "If you could live in any time period, which would you choose?",
                "What would you do if you were the CEO of a major company?",
                "Describe what happiness means to you.",
                "If you could master any skill instantly, what would it be?"
            ]
        }
        
        self.categories = list(self.prompts.keys())
        self.difficulty_levels = {
            'beginner': ['conversational', 'impromptu'],
            'intermediate': ['interview', 'presentation', 'academic'],
            'advanced': ['ielts_speaking', 'cat_preparation']
        }
    
    def get_random_prompt(self, category: str = None) -> str:
        """Get a random prompt from a specific category or all categories"""
        if category and category in self.prompts:
            return random.choice(self.prompts[category])
        else:
            # Choose from all categories
            all_prompts = []
            for prompts_list in self.prompts.values():
                all_prompts.extend(prompts_list)
            return random.choice(all_prompts)
    
    def get_prompts_by_category(self, category: str) -> List[str]:
        """Get all prompts from a specific category"""
        return self.prompts.get(category, [])
    
    def get_prompts_by_difficulty(self, difficulty: str) -> List[str]:
        """Get prompts by difficulty level"""
        if difficulty not in self.difficulty_levels:
            return []
        
        prompts = []
        for category in self.difficulty_levels[difficulty]:
            prompts.extend(self.prompts[category])
        
        return prompts
    
    def get_random_prompt_by_difficulty(self, difficulty: str) -> str:
        """Get a random prompt by difficulty level"""
        prompts = self.get_prompts_by_difficulty(difficulty)
        return random.choice(prompts) if prompts else self.get_random_prompt()
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return self.categories
    
    def get_difficulty_levels(self) -> List[str]:
        """Get all available difficulty levels"""
        return list(self.difficulty_levels.keys())
    
    def add_custom_prompt(self, category: str, prompt: str):
        """Add a custom prompt to a category"""
        if category not in self.prompts:
            self.prompts[category] = []
        
        self.prompts[category].append(prompt)
    
    def get_daily_challenge(self) -> Dict[str, str]:
        """Get a daily challenge with prompts from different categories"""
        challenge = {}
        
        # Get one prompt from each difficulty level
        for difficulty in self.difficulty_levels.keys():
            challenge[difficulty] = self.get_random_prompt_by_difficulty(difficulty)
        
        return challenge
    
    def get_themed_session(self, theme: str) -> List[str]:
        """Get a themed practice session with multiple related prompts"""
        themed_sessions = {
            'job_interview': [
                "Tell me about yourself and your background.",
                "What is your biggest strength?",
                "Describe a challenging situation you overcame.",
                "Where do you see yourself in 5 years?",
                "Why are you interested in this position?"
            ],
            'academic_presentation': [
                "Introduce your research topic.",
                "Explain your methodology.",
                "Present your key findings.",
                "Discuss the implications of your work.",
                "Address potential questions or criticisms."
            ],
            'business_pitch': [
                "Present your business idea.",
                "Explain the market opportunity.",
                "Describe your competitive advantage.",
                "Outline your financial projections.",
                "Ask for investment or support."
            ],
            'public_speaking': [
                "Introduce yourself to a new audience.",
                "Present a problem that needs solving.",
                "Propose your solution.",
                "Explain the benefits of your approach.",
                "Call your audience to action."
            ]
        }
        
        return themed_sessions.get(theme, [self.get_random_prompt() for _ in range(5)])
    
    def get_timed_challenges(self) -> Dict[str, List[str]]:
        """Get prompts organized by recommended speaking time"""
        return {
            '30_seconds': [
                "Introduce yourself briefly.",
                "What's your favorite hobby?",
                "Describe your morning routine.",
                "What's your biggest pet peeve?",
                "Name three things you're grateful for today."
            ],
            '1_minute': [
                "Describe your ideal vacation.",
                "What skill would you like to learn?",
                "Talk about a book you recently read.",
                "Explain your favorite recipe.",
                "Describe your dream job."
            ],
            '2_minutes': [
                "Tell me about a challenge you overcame.",
                "Describe a person who influenced your life.",
                "Explain a complex topic you understand well.",
                "Present a solution to a common problem.",
                "Share a meaningful life experience."
            ],
            '3_minutes': [
                "Present your thoughts on a current event.",
                "Describe your career journey so far.",
                "Explain how you would improve your community.",
                "Present a business idea you have.",
                "Discuss the future of your industry."
            ]
        }
    
    def get_prompt_with_context(self, category: str = None) -> Dict[str, str]:
        """Get a prompt with additional context and tips"""
        prompt = self.get_random_prompt(category)
        
        context_tips = {
            'interview': {
                'context': 'You are in a job interview. Be professional, confident, and specific.',
                'tips': 'Use the STAR method (Situation, Task, Action, Result) for behavioral questions.'
            },
            'academic': {
                'context': 'You are presenting to an academic audience. Be clear, evidence-based, and thorough.',
                'tips': 'Structure your response with clear introduction, body, and conclusion.'
            },
            'presentation': {
                'context': 'You are giving a presentation. Engage your audience and be persuasive.',
                'tips': 'Use storytelling, examples, and clear transitions between points.'
            },
            'conversational': {
                'context': 'You are having a casual conversation. Be natural, engaging, and personable.',
                'tips': 'Share personal experiences and ask follow-up questions.'
            },
            'ielts_speaking': {
                'context': 'This is an IELTS speaking test. Speak clearly and use varied vocabulary.',
                'tips': 'Aim for 1-2 minutes. Use linking words and express opinions clearly.'
            },
            'cat_preparation': {
                'context': 'This is a business school interview. Show leadership potential and business acumen.',
                'tips': 'Demonstrate analytical thinking and relate to business concepts.'
            }
        }
        
        # Determine category if not provided
        if not category:
            for cat, prompts in self.prompts.items():
                if prompt in prompts:
                    category = cat
                    break
        
        context_info = context_tips.get(category, {
            'context': 'Speak clearly and confidently.',
            'tips': 'Structure your response with clear beginning, middle, and end.'
        })
        
        return {
            'prompt': prompt,
            'category': category,
            'context': context_info['context'],
            'tips': context_info['tips']
        }
