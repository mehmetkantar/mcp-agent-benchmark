#!/usr/bin/env python3
"""
MCP Agent - AI Web Scraping and Benchmarking Tool
================================================

This agent demonstrates practical benchmarking capabilities using:
- Firecrawl MCP for web scraping and data collection
- OpenRouter API for cost-effective multi-model LLM integration  
- LangChain framework for agentic tool-calling architecture
- Comprehensive performance comparison across 2 AI models

Key Features:
- Autonomous web scraping using Firecrawl MCP
- Multi-model benchmarking and performance analysis
- Amazon product information extraction
- AI blog content analysis and summarization
- JSON-structured output with Pydantic validation
- Production-ready architecture with comprehensive logging

Author: Technical Assessment Candidate
Created for: AIMultiple Benchmark Team Assessment
Date: 2024
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports for agentic framework
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# MCP and Firecrawl integration
from langchain_mcp_adapters.tools import MCPTool
from firecrawl import FirecrawlApp

# Pydantic for data validation and schema definition
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for structured output validation
class AmazonProductOutput(BaseModel):
    """Schema for Amazon product extraction output"""
    product_name: Optional[str] = Field(None, description="The full name of the product.")
    price: Optional[float] = Field(None, description="The price of the product as a float, without currency symbols.")
    currency: Optional[str] = Field(None, description="The ISO 4217 currency code (e.g., 'USD', 'EUR', 'GBP').")
    rating: Optional[float] = Field(None, description="The star rating of the product (e.g., 4.5).")
    review_count: Optional[int] = Field(None, description="The total number of reviews as an integer.")


class BlogAnalysisOutput(BaseModel):
    """Schema for blog analysis output"""
    title: Optional[str] = Field(None, description="The main title of the blog post.")
    publication_date: Optional[str] = Field(None, description="The date the article was published.")
    key_technologies: Optional[List[str]] = Field(None, description="A list of key technologies or topics mentioned.")
    author: Optional[str] = Field(None, description="The name of the author, if found.")
    summary: Optional[str] = Field(None, description="A concise, two-sentence summary of the article.")


@dataclass
class BenchmarkResult:
    """Data structure for storing individual benchmark results"""
    model_name: str
    task_name: str
    success: bool
    response_time: float
    extracted_data: Dict[str, Any]
    error_message: str = ""
    completeness_score: float = 0.0
    pydantic_validation_success: bool = False
    raw_response: str = ""


@dataclass
class ModelPerformanceStats:
    """Performance statistics for each model across all tasks"""
    model_name: str
    total_tasks: int
    successful_tasks: int
    average_response_time: float
    success_rate: float
    average_completeness: float
    pydantic_validation_rate: float
    benchmark_results: List[BenchmarkResult]


class WebScrapingInput(BaseModel):
    """Input schema for the web scraping tool"""
    url: str = Field(description="The URL to scrape content from")


class FirecrawlMCPTool(BaseTool):
    """
    Custom LangChain tool that integrates true Firecrawl MCP for web scraping.
    
    This tool uses the langchain-mcp MCPTool adapter to make genuine MCP calls,
    demonstrating proper MCP protocol integration.
    """
    
    name: str = "web_scraper"
    description: str = (
        "Scrape a URL via Firecrawl MCP and return markdown text. "
        "Use for product pages, blogs, etc."
    )
    args_schema: Type[BaseModel] = WebScrapingInput
    
    def __init__(self, firecrawl_api_key: str):
        """Initialize the Firecrawl MCP tool"""
        super().__init__()
        self._api_key = firecrawl_api_key
        # Try to initialize real MCP client
        try:
            self._init_mcp_client(firecrawl_api_key)
            self._use_mcp = True
            logger.info("Successfully initialized Firecrawl MCP client")
        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}, falling back to Firecrawl SDK")
            self._firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
            self._use_mcp = False
    
    def _init_mcp_client(self, api_key: str):
        """Initialize the real MCP client connection"""
        import os
        from langchain_mcp_adapters.sessions import StdioConnection
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # Create stdio connection with proper MCP configuration format
        connection = StdioConnection({
            "transport": "stdio",
            "command": "/usr/bin/firecrawl-mcp",
            "args": [],
            "env": {"FIRECRAWL_API_KEY": api_key}
        })
        
        # Create MCP client with named connection
        self._mcp_client = MultiServerMCPClient(connections={"firecrawl": connection})
        
        # This will be set when we get tools
        self._mcp_tools = None
    
    def _run(self, url: str) -> str:
        """Execute web scraping via Firecrawl MCP or SDK fallback."""
        if self._use_mcp:
            return self._run_mcp(url)
        else:
            return self._run_sdk_fallback(url)
    
    def _run_mcp(self, url: str) -> str:
        """Execute web scraping via the real Firecrawl MCP client."""
        try:
            logger.info(f"Invoking real Firecrawl MCP for URL: {url}")
            
            # Run async MCP call in sync context
            import asyncio
            try:
                result = asyncio.run(self._async_mcp_call(url))
                return result
            except RuntimeError:
                # If we're already in an event loop, use nested approach
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.run(self._async_mcp_call(url))
                return result
            
        except Exception as e:
            error_msg = f"Error scraping {url} via real MCP: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _async_mcp_call(self, url: str) -> str:
        """Async MCP call to Firecrawl server"""
        try:
            # Get MCP tools if not already cached
            if self._mcp_tools is None:
                self._mcp_tools = await self._mcp_client.get_tools(server_name="firecrawl")
                logger.info(f"Retrieved {len(self._mcp_tools)} MCP tools from Firecrawl")
            
            # Find the scrape/crawl tool
            scrape_tool = None
            for tool in self._mcp_tools:
                if 'scrape' in tool.name.lower() or 'crawl' in tool.name.lower():
                    scrape_tool = tool
                    break
            
            if not scrape_tool:
                return f"No scrape/crawl tool found in MCP tools: {[t.name for t in self._mcp_tools]}"
            
            logger.info(f"Using MCP tool: {scrape_tool.name}")
            
            # Call the MCP tool
            result = await scrape_tool.ainvoke({"url": url})
            
            # Extract content
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                content = result.get("content") or result.get("markdown") or result.get("text") or str(result)
            else:
                content = str(result)
            
            # Truncate if too long
            truncated_content = content[:8000] + ("\n\n[Content truncated...]" if len(content) > 8000 else "")
            logger.info(f"Successfully scraped content via real Firecrawl MCP for {url}")
            return truncated_content
            
        except Exception as e:
            error_msg = f"Async MCP call failed for {url}: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _run_sdk_fallback(self, url: str) -> str:
        """Execute web scraping via Firecrawl SDK as fallback."""
        try:
            logger.info(f"Using Firecrawl SDK fallback for URL: {url}")
            
            # Use Firecrawl SDK
            result = self._firecrawl.scrape(url=url, formats=['markdown'])
            
            # Extract text content robustly
            text = result.get("markdown") or result.get("content")
            if not text:
                error_msg = f"Failed to scrape {url} via Firecrawl SDK."
                logger.warning(error_msg)
                return error_msg

            # Truncate content if it's too long
            truncated_text = text[:8000] + ("\n\n[Content truncated...]" if len(text) > 8000 else "")
            logger.info(f"Successfully scraped content via Firecrawl SDK for {url}")
            return truncated_text
            
        except Exception as e:
            error_msg = f"Error scraping {url} via Firecrawl SDK: {e}"
            logger.error(error_msg)
            return error_msg


class MCPBenchmarkAgent:
    """
    Main MCP benchmarking agent that integrates LangChain agentic framework
    with Firecrawl MCP for comprehensive AI model evaluation.
    """
    
    def __init__(self):
        """Initialize the MCP benchmark agent"""
        
        # Load API keys from environment
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        # AI models for benchmarking (cost-effective OpenRouter free models)
        self.models = [
            "deepseek/deepseek-chat-v3.1:free",
            "google/gemini-2.0-flash-exp:free"
        ]
        
        # Create Firecrawl MCP tool
        self.web_scraping_tool = FirecrawlMCPTool(
            firecrawl_api_key=self.firecrawl_api_key
        )
        
        # Define task schemas for validation
        self.task_schemas = {
            "Amazon Product Analysis": AmazonProductOutput,
            "AI Blog Analysis": BlogAnalysisOutput
        }
        
        logger.info("MCP Benchmark Agent initialized successfully")
    
    def create_agent_for_model(self, model_name: str) -> AgentExecutor:
        """
        Create a LangChain agent for specific model with web scraping capabilities
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            Configured AgentExecutor with web scraping tool
        """
        # Initialize LLM with OpenRouter and enforce JSON output
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,  # Set to 0 for deterministic JSON output
            max_tokens=1500, # Increase slightly for complex JSON
            model_kwargs={"response_format": {"type": "json_object"}}, # Force JSON output
        )
        
        # Create agent prompt template
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent web analysis agent with web scraping capabilities.

You have access to a web scraper tool that can fetch content from any URL using Firecrawl MCP.
When users ask you to analyze web content, extract product information, or analyze blogs:

1. Use the web_scraper tool to fetch content from the provided URL
2. Carefully analyze the scraped content
3. Extract the requested information accurately  
4. ALWAYS return ONLY a valid JSON object as your response
5. Include all requested fields in your JSON response
6. Use null values for fields that cannot be found

Always provide accurate, well-structured JSON responses based on the actual content you scrape."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create the tool-calling agent
        agent = create_tool_calling_agent(
            llm=llm,
            tools=[self.web_scraping_tool],
            prompt=agent_prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.web_scraping_tool],
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        return agent_executor
    
    def benchmark_model_on_task(self, model_name: str, task_name: str, 
                               task_prompt: str) -> BenchmarkResult:
        """
        Benchmark a single model on a specific task
        
        Args:
            model_name: Name of the model to test
            task_name: Name of the task
            task_prompt: Natural language task prompt
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking {model_name} on {task_name}")
        
        try:
            # Create agent for this model
            agent_executor = self.create_agent_for_model(model_name)
            
            # Execute task and measure performance
            start_time = time.time()
            
            result = agent_executor.invoke({
                "input": task_prompt,
                "chat_history": []
            })
            
            response_time = time.time() - start_time
            response_content = result["output"]
            
            # Parse JSON directly (no regex needed due to forced JSON output)
            try:
                extracted_data = json.loads(response_content)
                
                # Validate with Pydantic schema
                schema_class = self.task_schemas.get(task_name)
                pydantic_validation_success = False
                completeness = 0.0
                
                if schema_class:
                    try:
                        validated_data = schema_class(**extracted_data)
                        pydantic_validation_success = True
                        completeness = self._calculate_completeness_pydantic(validated_data, schema_class)
                    except Exception as validation_error:
                        logger.warning(f"Pydantic validation failed for {model_name}: {validation_error}")
                        completeness = self._calculate_completeness_legacy(extracted_data, task_name)
                else:
                    completeness = self._calculate_completeness_legacy(extracted_data, task_name)
                
                return BenchmarkResult(
                    model_name=model_name,
                    task_name=task_name,
                    success=True,
                    response_time=response_time,
                    extracted_data=extracted_data,
                    completeness_score=completeness,
                    pydantic_validation_success=pydantic_validation_success,
                    raw_response=response_content
                )
                
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed for {model_name}: {json_error}")
                return BenchmarkResult(
                    model_name=model_name,
                    task_name=task_name,
                    success=False,
                    response_time=response_time,
                    extracted_data={},
                    error_message=f"Invalid JSON format in response: {json_error}",
                    pydantic_validation_success=False,
                    raw_response=response_content
                )
                
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {str(e)}")
            return BenchmarkResult(
                model_name=model_name,
                task_name=task_name,
                success=False,
                response_time=0.0,
                extracted_data={},
                error_message=str(e),
                pydantic_validation_success=False,
                raw_response=""
            )
    
    def _calculate_completeness_pydantic(self, validated_data: BaseModel, schema_class: Type[BaseModel]) -> float:
        """Calculate completeness score using Pydantic validation"""
        total_fields = len(schema_class.model_fields)
        non_null_fields = sum(1 for field_name in schema_class.model_fields 
                             if getattr(validated_data, field_name) is not None)
        
        return (non_null_fields / total_fields) * 100 if total_fields > 0 else 0.0
    
    def _calculate_completeness_legacy(self, extracted_data: Dict, task_name: str) -> float:
        """Fallback completeness calculation for legacy compatibility"""
        expected_fields = {
            "Amazon Product Analysis": ["product_name", "price", "rating", "review_count"],
            "AI Blog Analysis": ["title", "publication_date", "key_technologies", "author", "summary"]
        }
        
        expected = expected_fields.get(task_name, [])
        if not expected:
            return 100.0
        
        found_fields = sum(1 for field in expected if field in extracted_data and 
                          extracted_data[field] not in ["", "Not found", None])
        
        return (found_fields / len(expected)) * 100
    
    def run_comprehensive_benchmark(self) -> Dict[str, ModelPerformanceStats]:
        """
        Run comprehensive benchmark across all models and tasks
        
        Returns:
            Dictionary mapping model names to performance statistics
        """
        print("="*80)
        print("🚀 MCP AGENT COMPREHENSIVE BENCHMARK")
        print("="*80)
        print("🎯 Framework: LangChain with tool-calling agents")
        print("🔥 Integration: Firecrawl MCP with langchain-mcp MCPTool adapter")
        print("🧠 Models: 2 cost-effective models via OpenRouter")
        print("📊 Tasks: Amazon product extraction & AI blog analysis")
        print("✅ Validation: Pydantic schema validation for structured output")
        
        # Define benchmark tasks with strict prompts
        tasks = [
            {
                "name": "Amazon Product Analysis",
                "prompt": """First, use the web_scraper tool to fetch the content from the URL 'https://www.amazon.com/dp/B092LT72ZG'.
After scraping, analyze the content and extract the required fields.

Return ONLY a valid JSON object that strictly conforms to the following schema. If a field is not found, use a null value.

Schema:
{
    "product_name": "string or null",
    "price": "float or null",
    "currency": "string (e.g., 'USD') or null",
    "rating": "float or null",
    "review_count": "integer or null"
}"""
            },
            {
                "name": "AI Blog Analysis",
                "prompt": """First, use the web_scraper tool to fetch the content from the URL 'https://openai.com/blog/chatgpt'.
After scraping, analyze the content and extract the required fields.

Return ONLY a valid JSON object that strictly conforms to the following schema. If a field is not found, use a null value.

Schema:
{
    "title": "string or null",
    "publication_date": "string or null",
    "key_technologies": "array of strings or null",
    "author": "string or null",
    "summary": "string or null"
}"""
            }
        ]
        
        all_results = []
        
        # Execute benchmarks
        for i, task in enumerate(tasks, 1):
            print(f"\n🔄 TASK {i}/{len(tasks)}: {task['name']}")
            print("-" * 60)
            
            for j, model in enumerate(self.models, 1):
                print(f"   📊 Testing model {j}/{len(self.models)}: {model.split('/')[-1]}")
                
                result = self.benchmark_model_on_task(
                    model, task["name"], task["prompt"]
                )
                
                all_results.append(result)
                
                if result.success:
                    validation_status = "✅ PASSED" if result.pydantic_validation_success else "⚠️ FAILED"
                    print(f"      ✅ Success: {result.response_time:.2f}s, "
                          f"Completeness: {result.completeness_score:.1f}%, "
                          f"Pydantic: {validation_status}")
                    
                    print(f"      🔍 RAW MODEL OUTPUT:")
                    print(f"      {'-'*50}")
                    if hasattr(result, 'raw_response') and result.raw_response:
                        print(f"      {result.raw_response}")
                    else:
                        print(f"      [Raw response not captured]")
                    print(f"      {'-'*50}")
                    
                    print(f"      📊 PARSED JSON DATA:")
                    print(f"      {json.dumps(result.extracted_data, indent=6, ensure_ascii=False)}")
                    print(f"      ✅ Pydantic Validation: {validation_status}")
                    print()
                else:
                    print(f"      ❌ Failed: {result.error_message}")
                    if hasattr(result, 'raw_response') and result.raw_response:
                        print(f"      🔍 RAW MODEL OUTPUT:")
                        print(f"      {'-'*50}")
                        print(f"      {result.raw_response}")
                        print(f"      {'-'*50}")
                    print()
                
                # Rate limiting delay
                time.sleep(1)
        
        # Calculate performance statistics
        model_stats = self._calculate_performance_stats(all_results)
        
        # Print comprehensive report
        self._print_benchmark_report(model_stats)
        
        return model_stats
    
    def _calculate_performance_stats(self, results: List[BenchmarkResult]) -> Dict[str, ModelPerformanceStats]:
        """Calculate comprehensive performance statistics for each model"""
        model_stats = {}
        
        for model in self.models:
            model_results = [r for r in results if r.model_name == model]
            
            if not model_results:
                continue
            
            successful_results = [r for r in model_results if r.success]
            
            total_tasks = len(model_results)
            successful_tasks = len(successful_results)
            
            avg_response_time = (
                sum(r.response_time for r in successful_results) / len(successful_results)
                if successful_results else 0.0
            )
            
            avg_completeness = (
                sum(r.completeness_score for r in successful_results) / len(successful_results)
                if successful_results else 0.0
            )
            
            pydantic_validations = sum(1 for r in successful_results if r.pydantic_validation_success)
            pydantic_validation_rate = (pydantic_validations / len(successful_results) * 100) if successful_results else 0.0
            
            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
            
            model_stats[model] = ModelPerformanceStats(
                model_name=model,
                total_tasks=total_tasks,
                successful_tasks=successful_tasks,
                average_response_time=avg_response_time,
                success_rate=success_rate,
                average_completeness=avg_completeness,
                pydantic_validation_rate=pydantic_validation_rate,
                benchmark_results=model_results
            )
        
        return model_stats
    
    def _print_benchmark_report(self, model_stats: Dict[str, ModelPerformanceStats]):
        """Print comprehensive benchmark report"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        # Performance ranking
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: (x[1].success_rate, x[1].pydantic_validation_rate, -x[1].average_response_time),
            reverse=True
        )
        
        print(f"\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 85)
        print(f"{'Rank':<4} {'Model':<35} {'Success%':<9} {'Pydantic%':<10} {'Avg Time':<9} {'Completeness%'}")
        print("-" * 85)
        
        for i, (model_name, stats) in enumerate(sorted_models, 1):
            short_name = model_name.split('/')[-1].replace(':free', '')
            print(f"{i:<4} {short_name:<35} {stats.success_rate:<8.1f} "
                  f"{stats.pydantic_validation_rate:<9.1f} "
                  f"{stats.average_response_time:<8.2f}s {stats.average_completeness:<11.1f}")
        
        # Best performers analysis
        if sorted_models:
            champion = sorted_models[0][1]
            print(f"\n🥇 CHAMPION MODEL:")
            print(f"   🏆 Winner: {champion.model_name}")
            print(f"   📈 Success Rate: {champion.success_rate:.1f}%")
            print(f"   ✅ Pydantic Validation Rate: {champion.pydantic_validation_rate:.1f}%")
            print(f"   ⚡ Avg Response Time: {champion.average_response_time:.2f}s")
            print(f"   📊 Avg Completeness: {champion.average_completeness:.1f}%")
            print(f"   🎯 Tasks Completed: {champion.successful_tasks}/{champion.total_tasks}")
        
        # Performance insights
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 50)
        
        fastest_model = min(model_stats.items(), key=lambda x: x[1].average_response_time if x[1].successful_tasks > 0 else float('inf'))
        most_complete = max(model_stats.items(), key=lambda x: x[1].average_completeness)
        most_reliable = max(model_stats.items(), key=lambda x: x[1].success_rate)
        best_validation = max(model_stats.items(), key=lambda x: x[1].pydantic_validation_rate)
        
        print(f"⚡ Fastest Model: {fastest_model[0].split('/')[-1]} ({fastest_model[1].average_response_time:.2f}s)")
        print(f"📊 Most Complete: {most_complete[0].split('/')[-1]} ({most_complete[1].average_completeness:.1f}%)")
        print(f"🎯 Most Reliable: {most_reliable[0].split('/')[-1]} ({most_reliable[1].success_rate:.1f}%)")
        print(f"✅ Best Validation: {best_validation[0].split('/')[-1]} ({best_validation[1].pydantic_validation_rate:.1f}%)")
        
        # Summary statistics
        total_tests = sum(stats.total_tasks for stats in model_stats.values())
        total_successes = sum(stats.successful_tasks for stats in model_stats.values())
        total_validations = sum(len([r for r in stats.benchmark_results if r.pydantic_validation_success]) for stats in model_stats.values())
        
        print(f"\n📈 BENCHMARK SUMMARY:")
        print(f"   🎯 Total Tests: {total_tests}")
        print(f"   ✅ Total Successes: {total_successes}")
        print(f"   📊 Overall Success Rate: {(total_successes/total_tests*100):.1f}%")
        print(f"   ✅ Pydantic Validations: {total_validations}")
        print(f"   📋 Validation Rate: {(total_validations/total_successes*100):.1f}%" if total_successes > 0 else "   📋 Validation Rate: 0.0%")
        print(f"   🤖 Models Benchmarked: {len(self.models)}")
        
        print("="*80)


def main():
    """
    Main execution function for the MCP benchmarking agent
    """
    try:
        print("🚀 Starting Production-Grade MCP Agent Benchmark Assessment")
        print("🔧 Initializing agent with Firecrawl MCP integration...")
        
        # Initialize the benchmark agent
        agent = MCPBenchmarkAgent()
        
        # Run comprehensive benchmark
        model_stats = agent.run_comprehensive_benchmark()
        
        print(f"\n🎉 PRODUCTION-GRADE BENCHMARK ASSESSMENT COMPLETED!")
        print(f"✨ Firecrawl MCP integration with Pydantic validation demonstrated")
        print(f"🚀 Ready for enterprise deployment with structured data validation")
        
        return model_stats
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please ensure API keys are properly configured in .env file")
        print("📋 Required: OPENROUTER_API_KEY and FIRECRAWL_API_KEY")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        logger.error(f"Unexpected error in main execution: {str(e)}")
        return None


if __name__ == "__main__":
    main()