import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2
import zipfile
from docx import Document
from io import BytesIO


class DocumentParser:
    @staticmethod
    def extract_text(file_obj):
        """
        Extract text from DOCX/PDF files.
        Handles corrupted DOCX files gracefully.
        """
        file_obj.seek(0)
        
        # Handle both file objects and file paths
        file_name = getattr(file_obj, 'name', '')
        if not file_name and hasattr(file_obj, 'filename'):
            file_name = file_obj.filename
        
        if file_name.endswith('.docx'):
            return DocumentParser._extract_docx(file_obj)
        elif file_name.endswith('.pdf'):
            return DocumentParser._extract_pdf(file_obj)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")

    @staticmethod
    def _extract_docx(file_obj):
        """Extract text from DOCX with corruption handling"""
        try:
            doc = Document(file_obj)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
            
        except (zipfile.BadZipFile, Exception) as e:
            print(f"Standard DOCX parsing failed ({e}), trying fallback...")
            return DocumentParser._extract_docx_fallback(file_obj)

    @staticmethod
    def _extract_docx_fallback(file_obj):
        """Manual extraction from corrupted DOCX ZIP structure"""
        try:
            file_obj.seek(0)
            content = file_obj.read()
            
            with zipfile.ZipFile(BytesIO(content)) as zf:
                xml_data = zf.read('word/document.xml')
                
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_data)
                
                texts = []
                for elem in root.iter():
                    if elem.tag.endswith('}t') and elem.text:
                        texts.append(elem.text)
                
                full_text = ''.join(texts)
                # Better formatting: split on sentence endings
                import re
                formatted = re.sub(r'([.!?])(\w)', r'\1\n\2', full_text)
                return formatted
                
        except Exception as e:
            raise ValueError(f"Could not extract from corrupted DOCX: {e}")

    @staticmethod
    def _extract_pdf(file_obj):
        """Extract text from PDF using PyPDF2"""
        try:
            file_obj.seek(0)
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Could not extract PDF: {e}")


class TextPreprocessor:
    """Preprocess text for matching"""

    DEFAULT_SKILLS = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 
        'go', 'golang', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r',
        'matlab', 'perl', 'shell scripting', 'bash', 'powershell', 'dart',
        'objective-c', 'objective c', 'vba', 'abap', 'cobol', 'fortran', 
        'lua', 'julia', 'haskell', 'elixir', 'erlang', 'clojure', 'f#',
        'groovy', 'visual basic', 'vb.net', 'delphi', 'pascal', 'prolog',
        'lisp', 'scheme', 'tcl', 'awk', 'sed', 'solidity', 'vyper',
        'assembly', 'asm', 'mips', 'arm', 'verilog', 'vhdl', 'systemverilog',
        'cuda', 'opencl', 'glsl', 'hlsl', 'shader programming',
        
        # Web Development
        'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less', 'tailwind css',
        'bootstrap', 'material ui', 'mui', 'chakra ui', 'antd',
        'react', 'react.js', 'reactjs', 'next.js', 'nextjs', 'gatsby',
        'angular', 'angularjs', 'vue', 'vue.js', 'vuejs', 'nuxt.js', 'nuxtjs',
        'svelte', 'sveltekit', 'solid.js', 'preact', 'alpine.js',
        'jquery', 'backbone.js', 'ember.js', 'knockout.js',
        'node.js', 'nodejs', 'express', 'express.js', 'nestjs', 'fastify',
        'koa', 'hapi', 'sails.js', 'meteor', 'feathers.js',
        'django', 'flask', 'fastapi', 'tornado', 'bottle', 'pyramid',
        'spring boot', 'spring framework', 'spring mvc', 'spring security',
        'asp.net', 'asp.net core', '.net core', '.net framework', 'blazor',
        'laravel', 'symfony', 'codeigniter', 'cakephp', 'yii', 'zend',
        'rails', 'ruby on rails', 'sinatra', 'hanami',
        'wordpress', 'drupal', 'joomla', 'magento', 'shopify', 'woocommerce',
        'web design', 'responsive design', 'progressive web apps', 'pwa',
        'web components', 'webassembly', 'wasm', 'websockets', 'socket.io',
        'web rtc', 'webrtc', 'server-side rendering', 'ssr', 'static site generation',
        
        # Mobile Development
        'android', 'android sdk', 'android studio', 'android jetpack',
        'ios', 'iphone development', 'ipad development', 'swiftui', 'uikit',
        'react native', 'flutter', 'xamarin', 'ionic', 'cordova',
        'phonegap', 'capacitor', 'native script', 'titanium',
        'mobile development', 'mobile app development', 'cross-platform development',
        'app store optimization', 'aso', 'mobile ui design', 'mobile ux',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'mariadb',
        'mongodb', 'mongoose', 'cassandra', 'couchdb', 'couchbase',
        'redis', 'memcached', 'elasticsearch', 'solr', 'opensearch',
        'dynamodb', 'firebase', 'firestore', 'cosmos db', 'cockroachdb',
        'neo4j', 'arangodb', 'orientdb', 'graph database', 'influxdb',
        'timescaledb', 'clickhouse', 'snowflake',
        'bigquery', 'redshift', 'amazon rds', 'amazon aurora',
        'oracle', 'oracle database', 'pl/sql', 't-sql', 'transact-sql',
        'database design', 'database development', 'database administration',
        'dba', 'database optimization', 'query optimization', 'indexing',
        'data modeling', 'erd', 'entity relationship diagram', 'normalization',
        'acid', 'cap theorem', 'sharding', 'replication', 'database migration',
        'oltp', 'olap', 'data warehousing', 'etl', 'elt',
        
        # Cloud & DevOps
        'aws', 'amazon web services', 'ec2', 's3', 'lambda', 'rds', 'ecs',
        'eks', 'fargate', 'elastic beanstalk', 'cloudfront', 'route 53',
        'api gateway', 'sqs', 'sns', 'eventbridge', 'step functions',
        'azure', 'microsoft azure', 'azure devops', 'azure functions',
        'azure app service', 'azure kubernetes service', 'aks',
        'gcp', 'google cloud platform', 'google cloud', 'compute engine',
        'app engine', 'cloud run', 'cloud functions', 'bigquery',
        'ibm cloud', 'oracle cloud', 'alibaba cloud', 'digitalocean',
        'linode', 'vultr', 'heroku', 'netlify', 'vercel', 'render',
        'docker', 'docker compose', 'dockerfile', 'docker swarm',
        'kubernetes', 'k8s', 'helm', 'istio', 'linkerd', 'envoy',
        'jenkins', 'github actions', 'gitlab ci', 'gitlab ci/cd',
        'circleci', 'travis ci', 'bamboo', 'teamcity', 'azure pipelines',
        'argo cd', 'argo workflows', 'flux cd', 'spinnaker',
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
        'ci/cd', 'continuous integration', 'continuous deployment',
        'continuous delivery', 'devops', 'devsecops', 'gitops',
        'terraform', 'pulumi', 'cloudformation', 'ansible', 'chef',
        'puppet', 'saltstack', 'vagrant', 'packer', 'vault',
        'linux', 'ubuntu', 'debian', 'centos', 'rhel', 'fedora',
        'arch linux', 'alpine linux', 'unix', 'bsd', 'freebsd',
        'windows server', 'windows administration', 'active directory',
        'nginx', 'apache', 'apache tomcat', 'iis', 'caddy',
        'haproxy', 'traefik', 'varnish', 'cdn',
        'prometheus', 'grafana', 'elk stack', 'elasticsearch logstash kibana',
        'splunk', 'datadog', 'new relic', 'dynatrace', 'appdynamics',
        'nagios', 'zabbix', 'pagerduty', 'opsgenie',
        'system administration', 'sysadmin', 'site reliability engineering',
        'sre', 'infrastructure as code', 'iac', 'configuration management',
        'load balancing', 'high availability', 'ha', 'disaster recovery',
        'backup solutions', 'network administration', 'firewall management',
        
        # Data Science & AI/ML
        'machine learning', 'ml', 'deep learning', 'dl', 'ai', 'artificial intelligence',
        'neural networks', 'cnn', 'rnn', 'lstm', 'gru', 'transformers',
        'attention mechanism', 'bert', 'gpt', 'llm', 'large language models',
        'generative ai', 'genai', 'stable diffusion', 'midjourney',
        'reinforcement learning', 'rl', 'supervised learning', 'unsupervised learning',
        'semi-supervised learning', 'self-supervised learning', 'few-shot learning',
        'transfer learning', 'fine-tuning', 'prompt engineering',
        'tensorflow', 'pytorch', 'keras', 'jax', 'flax', 'hugging face',
        'transformers library', 'langchain', 'llamaindex', 'openai api',
        'scikit-learn', 'sklearn', 'xgboost', 'lightgbm', 'catboost',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'bokeh', 'altair', 'dash', 'streamlit', 'gradio',
        'data analysis', 'data analytics', 'data science', 'data engineering',
        'data mining', 'data visualization', 'business intelligence', 'bi',
        'statistical analysis', 'statistics', 'probability', 'hypothesis testing',
        'a/b testing', 'experimental design', 'regression analysis',
        'time series analysis', 'forecasting', 'predictive modeling',
        'feature engineering', 'feature selection', 'dimensionality reduction',
        'pca', 't-sne', 'umap', 'clustering', 'k-means', 'dbscan',
        'anomaly detection', 'outlier detection', 'recommendation systems',
        'collaborative filtering', 'content-based filtering',
        'nlp', 'natural language processing', 'text mining', 'text analytics',
        'sentiment analysis', 'named entity recognition', 'ner',
        'part-of-speech tagging', 'pos tagging', 'topic modeling', 'lda',
        'word embeddings', 'word2vec', 'glove', 'fasttext',
        'computer vision', 'cv', 'image processing', 'image recognition',
        'object detection', 'yolo', 'rcnn', 'mask rcnn', 'segmentation',
        'ocr', 'optical character recognition', 'facial recognition',
        'speech recognition', 'asr', 'text-to-speech', 'tts',
        'big data', 'apache spark', 'spark', 'pyspark', 'hadoop',
        'mapreduce', 'hive', 'pig', 'kafka', 'apache kafka', 'confluent',
        'airflow', 'apache airflow', 'prefect', 'dagster', 'dbt',
        'data build tool', 'great expectations', 'monte carlo',
        'mlflow', 'kubeflow', 'weights & biases', 'wandb',
        'tensorboard', 'optuna', 'ray', 'dask', 'modin',
        'tableau', 'power bi', 'looker', 'qlik', 'sisense',
        'spss', 'sas', 'stata', 'minitab', 'jupyter', 'jupyter notebook',
        'google colab', 'kaggle', 'rapidminer', 'knime',
        
        # Cybersecurity
        'cybersecurity', 'information security', 'infosec', 'network security',
        'application security', 'appsec', 'cloud security',
        'penetration testing', 'pen testing', 'ethical hacking',
        'vulnerability assessment', 'vulnerability management',
        'threat modeling', 'threat intelligence', 'siem',
        'security operations center', 'soc', 'incident response',
        'digital forensics', 'malware analysis', 'reverse engineering',
        'owasp', 'secure coding', 'code review', 'static analysis',
        'dynamic analysis', 'sast', 'dast', 'iast', 'sca',
        'identity and access management', 'iam', 'sso', 'oauth',
        'openid connect', 'saml', 'mfa', 'multi-factor authentication',
        'encryption', 'cryptography', 'ssl/tls', 'pki', 'public key infrastructure',
        'firewall', 'ids', 'ips', 'intrusion detection', 'intrusion prevention',
        'vpn', 'zero trust', 'nist', 'iso 27001', 'soc 2', 'gdpr',
        'hipaa', 'pci dss', 'compliance', 'risk assessment',
        'kali linux', 'metasploit', 'burp suite', 'wireshark', 'nmap',
        'nessus', 'qualys', 'rapid7', 'crowdstrike', 'sentinelone',
        'palo alto networks', 'fortinet', 'check point', 'cisco security',
        
        # Networking
        'networking', 'network engineering',
        'tcp/ip', 'dns', 'dhcp', 'http', 'https', 'ftp', 'sftp', 'ssh',
        'vlan', 'subnetting', 'routing', 'ospf', 'bgp', 'eigrp',
        'mpls', 'sd-wan', 'ipsec',
        'wireless networking', 'wifi', 'wlan', 'bluetooth', 'zigbee',
        'network protocols', 'network architecture', 'network design',
        'cisco', 'cisco ios', 'cisco ccna', 'cisco ccnp', 'cisco ccie',
        'juniper', 'arista', 'f5', 'f5 load balancer', 'citrix netscaler',
        
        # Software Engineering
        'software engineering', 'software development', 'software architecture',
        'software design', 'system design', 'system architecture',
        'microservices', 'service-oriented architecture', 'soa',
        'event-driven architecture', 'eda', 'domain-driven design', 'ddd',
        'clean architecture', 'hexagonal architecture', 'onion architecture',
        'layered architecture', 'monolithic architecture', 'serverless architecture',
        'api development', 'api design', 'rest api', 'restful api', 'graphql',
        'grpc', 'protobuf', 'soap', 'xml rpc', 'json rpc', 'openapi',
        'swagger', 'postman', 'insomnia', 'api management',
        'webhooks', 'event sourcing', 'cqrs', 'saga pattern',
        'circuit breaker', 'bulkhead pattern', 'retry pattern', 'timeout pattern',
        'design patterns', 'singleton', 'factory', 'observer', 'strategy',
        'mvc', 'mvvm', 'mvp', 'clean code', 'solid principles',
        'dry principle', 'kiss principle', 'yagni',
        'test-driven development', 'tdd',
        'behavior-driven development', 'bdd',
        'extreme programming', 'xp', 'kanban', 'lean', 'safe',
        'agile', 'scrum', 'sprint planning', 'backlog grooming', 'retrospective',
        'jira', 'confluence', 'trello', 'asana', 'monday.com', 'linear',
        'version control', 'git flow', 'github flow', 'trunk-based development',
        'semantic versioning', 'conventional commits', 'changelog management',
        'software documentation', 'technical writing', 'api documentation',
        'readme', 'wiki', 'mkdocs', 'docusaurus', 'storybook',
        'oop', 'object-oriented programming', 'functional programming', 'fp',
        'procedural programming', 'declarative programming', 'imperative programming',
        'reactive programming', 'rxjava', 'rxjs', 'reactive streams',
        'concurrency', 'parallel programming', 'multithreading', 'async programming',
        'event loop', 'callbacks', 'promises', 'async/await', 'futures',
        'memory management', 'garbage collection', 'performance optimization',
        'profiling', 'debugging', 'unit testing', 'integration testing',
        'e2e testing', 'end-to-end testing', 'regression testing',
        'load testing', 'stress testing', 'performance testing',
        'security testing', 'usability testing', 'accessibility testing', 'a11y',
        'selenium', 'cypress', 'playwright', 'puppeteer', 'webdriver',
        'jest', 'mocha', 'chai', 'jasmine', 'karma', 'vitest',
        'pytest', 'unittest', 'nose', 'robot framework', 'cucumber',
        'test automation', 'qa automation', 'quality assurance', 'qa',
        'manual testing', 'black box testing', 'white box testing', 'grey box testing',
        'mutation testing', 'property-based testing', 'fuzz testing',
        'continuous testing', 'shift-left testing',
        
        # Project & Product Management
        'project management', 'program management', 'portfolio management',
        'product management', 'product owner', 'scrum master', 'agile coach',
        'stakeholder management', 'change management', 'risk management',
        'resource management', 'budget management', 'vendor management',
        'requirements gathering', 'business analysis', 'business analyst', 'ba',
        'user stories', 'use cases', 'user journey mapping', 'personas',
        'product roadmap', 'product strategy', 'go-to-market strategy',
        'market research', 'competitive analysis', 'swot analysis',
        'okrs', 'objectives and key results', 'kpis', 'metrics',
        'user acceptance testing', 'uat', 'beta testing',
        'product launch', 'feature flagging', 'feature toggles',
        'launchdarkly', 'split.io', 'optimizely', 'vwo',
        'pmi', 'pmp', 'prince2', 'itil', 'cobit', 'togaf',
        'microsoft project', 'smartsheet', 'notion', 'clickup',
        
        # UI/UX Design
        'ui design', 'ux design', 'ui/ux', 'user interface design',
        'user experience design', 'interaction design', 'ixd',
        'visual design', 'graphic design', 'motion design', 'animation',
        'wireframing', 'prototyping', 'mockups', 'storyboarding',
        'user research', 'heuristic evaluation',
        'accessibility design', 'inclusive design', 'design systems',
        'design thinking', 'human-centered design', 'hcd',
        'figma', 'sketch', 'adobe xd', 'invision', 'framer',
        'protopie', 'principle', 'after effects', 'premiere pro',
        'photoshop', 'illustrator', 'indesign', 'lightroom',
        'blender', 'cinema 4d', 'maya', '3ds max', 'zbrush',
        'color theory', 'typography', 'layout design', 'grid systems',
        'design tokens', 'component libraries', 'atomic design',
        
        # Game Development
        'game development', 'game design', 'game programming',
        'unity', 'unity3d', 'unreal engine', 'unreal engine 4', 'unreal engine 5',
        'godot', 'cryengine', 'game maker', 'construct',
        'cocos2d', 'phaser', 'pixi.js', 'three.js', 'babylon.js',
        'game physics', 'collision detection', 'pathfinding', 'a* algorithm',
        'game ai', 'npc behavior', 'procedural generation',
        'compute shaders',
        'level design', 'environment art', 'character art', 'concept art',
        'game audio', 'sound design', 'fmod', 'wwise',
        'multiplayer networking', 'game netcode', 'client-server architecture',
        'matchmaking', 'game analytics', 'monetization', 'in-app purchases',
        
        # Embedded Systems & IoT
        'embedded systems', 'embedded programming', 'firmware development',
        'microcontrollers', 'mcu', 'arduino', 'raspberry pi', 'esp32',
        'stm32', 'pic', 'avr', 'arm cortex', 'freescale', 'nxp',
        'real-time operating system', 'rtos', 'free rtos', 'zephyr',
        'embedded c', 'embedded c++', 'bare metal programming',
        'iot', 'internet of things', 'edge computing', 'mqtt',
        'coap', 'lora', 'lora wan', 'z-wave', 'thread',
        'matter protocol', 'home automation', 'industrial iot', 'iiot',
        'scada', 'plc programming', 'ladder logic', 'modbus',
        'can bus', 'lin bus', 'flexray', 'automotive electronics',
        'robotics', 'ros', 'robot operating system', 'slam',
        'computer vision for robotics', 'kinematics', 'inverse kinematics',
        'control systems', 'pid control', 'model predictive control',
        'signal processing', 'dsp', 'digital signal processing',
        'fpga', 'fpga design', 'xilinx', 'altera', 'intel fpga',
        'pcb design', 'altium designer', 'eagle', 'kicad', 'orcad',
        'circuit design', 'analog electronics', 'digital electronics',
        'power electronics', 'motor control', 'battery management',
        
        # Blockchain & Web3
        'blockchain', 'distributed ledger', 'smart contracts',
        'ethereum', 'web3.js', 'ethers.js',
        'hardhat', 'truffle', 'foundry', 'remix ide',
        'defi', 'decentralized finance', 'dex', 'amm',
        'nft', 'non-fungible tokens', 'erc-20', 'erc-721', 'erc-1155',
        'layer 2', 'rollups', 'optimistic rollups', 'zk rollups',
        'zero knowledge proofs', 'zk-snarks', 'zk-starks',
        'bitcoin', 'lightning network', 'hyperledger', 'fabric',
        'corda', 'quorum', 'polygon', 'arbitrum', 'optimism',
        'chainlink', 'the graph', 'ipfs', 'filecoin', 'swarm',
        'dao', 'decentralized autonomous organization',
        'tokenomics', 'consensus algorithms',
        'proof of work', 'proof of stake', 'delegated proof of stake',
        'byzantine fault tolerance', 'bft', 'pbft', 'raft',
        
        # Finance & Fintech
        'fintech', 'financial technology', 'digital banking',
        'payment systems', 'payment gateways', 'stripe', 'paypal',
        'square', 'adyen', 'braintree', 'razorpay', 'paytm',
        'banking software', 'core banking', 'loan management',
        'wealth management', 'robo-advisory',
        'trading systems', 'algorithmic trading', 'quantitative trading',
        'high frequency trading', 'hft', 'market making',
        'credit risk', 'market risk', 'operational risk',
        'regulatory reporting', 'basel', 'ifrs', 'gaap',
        'financial modeling', 'valuation', 'dcf',
        'bloomberg terminal', 'reuters', 'factset', 'morningstar',
        'actuarial science', 'insurance technology', 'insurtech',
        'cryptocurrency trading', 'forex trading', 'commodities trading',
        'blockchain finance', 'cbdc', 'central bank digital currency',
        
        # Healthcare & Biotech
        'health informatics', 'healthcare it', 'ehr', 'electronic health records',
        'emr', 'electronic medical records', 'hl7', 'fhir',
        'medical imaging', 'dicom', 'pacs', 'radiology informatics',
        'clinical decision support', 'cds', 'telemedicine', 'telehealth',
        'medical device software', 'fda validation', 'iec 62304',
        'hipaa compliance', 'gdpr healthcare', 'patient data privacy',
        'bioinformatics', 'computational biology', 'genomics', 'proteomics',
        'drug discovery', 'clinical trials', 'eclinical',
        'laboratory information system', 'lis', 'lims',
        'wearable technology', 'health monitoring', 'remote patient monitoring',
        'ai in healthcare', 'medical ai', 'diagnostic ai',
        
        # AR/VR/XR
        'augmented reality', 'ar', 'virtual reality', 'vr',
        'mixed reality', 'mr', 'extended reality', 'xr',
        'spatial computing', 'apple vision pro', 'meta quest',
        'oculus', 'htc vive', 'hololens', 'magic leap',
        'unity ar', 'unreal ar', 'arkit', 'arcore', 'vuforia',
        'webxr', 'openxr', 'steamvr', 'openvr',
        '3d modeling', '3d scanning', 'photogrammetry', 'lidar',
        'haptic feedback', 'motion tracking', 'hand tracking',
        'eye tracking', 'foveated rendering', 'passthrough',
        
        # Quantum Computing
        'quantum computing', 'quantum programming', 'qiskit',
        'cirq', 'pennylane', 'q#', 'quantum machine learning',
        'quantum algorithms', 'shor algorithm', 'grover algorithm',
        'quantum cryptography', 'quantum key distribution', 'qkd',
        'quantum error correction', 'superconducting qubits',
        'trapped ion', 'photonic quantum computing',
        
        # Legal & Compliance Tech
        'legal tech', 'contract management', 'clm',
        'e-discovery', 'ediscovery', 'legal research',
        'document automation', 'compliance management',
        'regtech', 'regulatory technology', 'aml', 'kyc',
        'anti-money laundering', 'know your customer',
        'data privacy', 'data protection',
        'cmmc', 'fedramp',
        
        # Education Tech
        'edtech', 'education technology', 'lms',
        'learning management system', 'moodle', 'canvas', 'blackboard',
        'd2l', 'brightspace', 'google classroom', 'microsoft teams education',
        'e-learning', 'online learning', 'blended learning',
        'mooc', 'massive open online course', 'coursera', 'udemy',
        'instructional design', 'curriculum development',
        'learning analytics', 'adaptive learning', 'gamification',
        'virtual classroom', 'webinar', 'scorm', 'xapi', 'lti',
        
        # Soft Skills & Methodologies
        'leadership', 'team leadership', 'people management',
        'mentoring', 'coaching', 'talent development',
        'communication', 'presentation skills', 'public speaking', 'storytelling',
        'problem solving', 'critical thinking', 'analytical thinking',
        'creativity', 'innovation',
        'negotiation', 'conflict resolution',
        'time management', 'prioritization', 'multitasking',
        'adaptability', 'flexibility', 'resilience',
        'emotional intelligence', 'eq', 'cultural awareness',
        'cross-functional collaboration', 'remote work', 'distributed teams',
        'english', 'french', 'spanish', 'german', 'mandarin', 'japanese',
        'technical translation', 'localization', 'l10n', 'internationalization', 'i18n',
        
        # Other Technical
        'distributed systems', 'event streaming', 'message queues', 'rabbitmq', 'activemq',
        'ibm mq', 'amazon sqs', 'google pub/sub', 'azure service bus',
        'protocol buffers', 'avro', 'thrift', 'messagepack',
        'json', 'xml', 'yaml', 'toml', 'csv', 'parquet', 'orc',
        'serialization', 'deserialization', 'data interchange',
        'regular expressions', 'regex', 'parsing', 'lexical analysis',
        'compiler design', 'interpreter design', 'language design',
        'virtual machines', 'jvm', 'clr', 'webassembly runtime',
        'operating systems', 'os development', 'kernel development',
        'device drivers', 'file systems',
        'containerization', 'virtualization', 'vmware', 'hyper-v',
        'kvm', 'xen', 'virtualbox', 'parallels', 'qemu',
        'service mesh', 'sidecar pattern', 'envoy proxy',
        'graphql federation', 'schema stitching', 'apollo federation',
        'trpc', 't3 stack', 'full stack development', 'mern stack',
        'mean stack', 'lamp stack', 'lemp stack', 'jamstack',
        'serverless', 'faas', 'function as a service',
        'edge functions', 'cdn edge',
        'web scraping', 'data scraping', 'screen scraping',
        'beautiful soup', 'scrapy', 'cheerio', 'requests', 'urllib',
        'disassembly', 'decompilation',
        'binary exploitation', 'buffer overflow',
        'format string attacks', 'rop chains', 'heap exploitation',
        'memory forensics', 'network forensics',
        'steganography', 'cryptanalysis', 'side-channel attacks',
        'hardware security', 'trusted platform module', 'tpm',
        'secure boot', 'trusted execution environment', 'tee',
        'confidential computing', 'homomorphic encryption',
    ]

    @staticmethod
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    @classmethod
    def extract_skills(cls, text, skill_list=None):
        if skill_list is None:
            skill_list = cls.DEFAULT_SKILLS

        text_lower = text.lower()
        found_skills = []

        for skill in skill_list:
            # Use word boundary matching for accuracy
            pattern = r'(?:^|\s)' + re.escape(skill.lower()) + r'(?:\s|$|[^a-z])'
            if re.search(pattern, text_lower):
                found_skills.append(skill)

        return found_skills


class MatchingEngine:
    """Main matching engine"""

    def __init__(self, use_ai=False):
        self.use_ai = use_ai
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        if self.use_ai:
            self.ai_model = AIModelInterface()
            self.ai_model.load_model()

    def calculate_similarity(self, resume_text, job_description):
        preprocessor = TextPreprocessor()
        clean_resume = preprocessor.preprocess(resume_text)
        clean_job = preprocessor.preprocess(job_description)

        if self.use_ai:
            return self._ai_matching(clean_resume, clean_job, resume_text, job_description)

        return self._tfidf_matching(clean_resume, clean_job, resume_text, job_description)

    def _extract_experience_years(self, text):
        """Extract years of experience mentioned in text"""
        import re
        # Look for patterns like "5 years", "3+ years", "2020 - 2024"
        patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d{4})\s*-\s*(?:present|current|\d{4})',
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(m) if isinstance(m, str) and m.isdigit() else m for m in matches])
        return years

    def _extract_education_level(self, text):
        """Detect education level from text"""
        education_keywords = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'masters': 4, 'mba': 4, 'msc': 4, 'ma': 4,
            'bachelors': 3, 'bs': 3, 'ba': 3, 'b.sc': 3, 'beng': 3,
            'associate': 2, 'diploma': 2,
            'high school': 1, 'secondary': 1
        }
        text_lower = text.lower()
        max_level = 0
        for keyword, level in education_keywords.items():
            if keyword in text_lower:
                max_level = max(max_level, level)
        return max_level

    def _tfidf_matching(self, clean_resume, clean_job, raw_resume, raw_job):
        try:
            # Use fit_transform on combined corpus for better results
            all_texts = [clean_job, clean_resume]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            similarity_score = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )[0][0]

            # Skills matching
            preprocessor = TextPreprocessor()
            resume_skills = preprocessor.extract_skills(raw_resume)
            job_skills = preprocessor.extract_skills(raw_job)

            if job_skills:
                matched_skills = [s for s in resume_skills if s in job_skills]
                missing_skills = [s for s in job_skills if s not in resume_skills]
                skills_score = len(matched_skills) / len(job_skills)
            else:
                matched_skills = resume_skills
                missing_skills = []
                skills_score = 0.0

            # Experience matching (NEW)
            resume_exp = self._extract_experience_years(raw_resume)
            job_exp = self._extract_experience_years(raw_job)
            if job_exp and resume_exp:
                # Simple scoring: any experience mentioned vs required
                exp_score = min(len(resume_exp), 2) / max(len(job_exp), 1)
            else:
                exp_score = 0.5  # Neutral if not specified

            # Education matching (NEW)
            resume_edu = self._extract_education_level(raw_resume)
            job_edu = self._extract_education_level(raw_job)
            if job_edu > 0 and resume_edu > 0:
                edu_score = min(resume_edu, job_edu) / max(job_edu, resume_edu)
            else:
                edu_score = 0.5  # Neutral if not specified

            # Weighted overall score
            overall_score = (
                similarity_score * 0.4 +      # Semantic similarity
                skills_score * 0.3 +           # Skills match
                exp_score * 0.2 +              # Experience match
                edu_score * 0.1                # Education match
            )

            return {
                'similarity_score': float(similarity_score),
                'skills_match_score': float(skills_score),
                'experience_match_score': float(exp_score),
                'education_match_score': float(edu_score),
                'overall_score': float(overall_score),
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
            }

        except Exception as e:
            return {
                'similarity_score': 0.0,
                'skills_match_score': 0.0,
                'experience_match_score': 0.0,
                'education_match_score': 0.0,
                'overall_score': 0.0,
                'matched_skills': [],
                'missing_skills': [],
                'error': str(e)
            }

    def _ai_matching(self, clean_resume, clean_job, raw_resume, raw_job):
        try:
            # Get semantic similarity from AI
            resume_emb, job_emb = self.ai_model.encode([clean_resume, clean_job])
            similarity = self.ai_model.calculate_similarity(resume_emb, job_emb)

            # Still use rule-based for skills/experience/education
            preprocessor = TextPreprocessor()
            resume_skills = preprocessor.extract_skills(raw_resume)
            job_skills = preprocessor.extract_skills(raw_job)

            if job_skills:
                matched_skills = [s for s in resume_skills if s in job_skills]
                missing_skills = [s for s in job_skills if s not in resume_skills]
                skills_score = len(matched_skills) / len(job_skills)
            else:
                matched_skills = resume_skills
                missing_skills = []
                skills_score = 0.0

            # Experience & Education
            exp_score = self._extract_experience_years(raw_resume)
            edu_score = self._extract_education_level(raw_resume)

            # AI-enhanced overall score
            overall_score = (
                float(similarity) * 0.5 +      # AI semantic similarity (higher weight)
                skills_score * 0.3 +
                (1.0 if exp_score else 0.5) * 0.1 +
                (1.0 if edu_score >= 3 else 0.5) * 0.1
            )

            return {
                'similarity_score': float(similarity),
                'skills_match_score': float(skills_score),
                'experience_match_score': float(exp_score) if isinstance(exp_score, (int, float)) else 0.5,
                'education_match_score': float(edu_score) if isinstance(edu_score, (int, float)) else 0.5,
                'overall_score': float(overall_score),
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
            }

        except Exception as e:
            return {
                'similarity_score': 0.0,
                'skills_match_score': 0.0,
                'experience_match_score': 0.0,
                'education_match_score': 0.0,
                'overall_score': 0.0,
                'matched_skills': [],
                'missing_skills': [],
                'error': str(e)
            }

    def rank_candidates(self, resumes_data, job_description):
        results = []

        for resume_data in resumes_data:
            match_result = self.calculate_similarity(
                resume_data['text'],
                job_description
            )
            match_result['resume_id'] = resume_data['id']
            match_result['applicant_name'] = resume_data.get('applicant_name', 'Unknown')
            results.append(match_result)

        results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        return results


class AIModelInterface:
    def __init__(self, model_name='sentence-bert'):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, texts):
        return self.model.encode(texts)

    def calculate_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
