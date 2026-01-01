"""
File export functionality - save transcripts and summaries in various formats
"""
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import OUTPUTS_DIR, EXPORT_SETTINGS
from src.logger import setup_logger

logger = setup_logger(__name__)


class FileExporter:
    """Export transcripts and summaries to various file formats"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def export_to_txt(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export content to a text file.
        
        Args:
            content: Text content to export
            filename: Output filename (without extension)
            metadata: Optional metadata to include at the top
        
        Returns:
            str: Path to exported file
        """
        try:
            output_path = self.output_dir / f"{filename}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add metadata header if provided
                if metadata and EXPORT_SETTINGS.get('include_metadata', True):
                    f.write(self._format_metadata_header(metadata))
                    f.write("\n" + "="*80 + "\n\n")
                
                # Write content
                f.write(content)
            
            logger.info(f"Exported to TXT: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error exporting to TXT: {str(e)}")
            raise
    
    def export_to_markdown(
        self,
        content: str,
        filename: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export content to a Markdown file.
        
        Args:
            content: Text content to export
            filename: Output filename (without extension)
            title: Optional title for the document
            metadata: Optional metadata to include
        
        Returns:
            str: Path to exported file
        """
        try:
            output_path = self.output_dir / f"{filename}.md"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add title
                if title:
                    f.write(f"# {title}\n\n")
                
                # Add metadata
                if metadata and EXPORT_SETTINGS.get('include_metadata', True):
                    f.write("## Metadata\n\n")
                    for key, value in metadata.items():
                        f.write(f"- **{key}**: {value}\n")
                    f.write("\n---\n\n")
                
                # Write content
                f.write(content)
            
            logger.info(f"Exported to Markdown: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error exporting to Markdown: {str(e)}")
            raise
    
    def export_to_json(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Dictionary data to export
            filename: Output filename (without extension)
        
        Returns:
            str: Path to exported file
        """
        try:
            output_path = self.output_dir / f"{filename}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported to JSON: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def export_complete_session(
        self,
        transcript_result: Dict[str, Any],
        summary_result: Dict[str, Any],
        base_filename: str
    ) -> Dict[str, str]:
        """
        Export complete transcription session (transcript + summary) in multiple formats.
        
        Args:
            transcript_result: Transcription result dictionary
            summary_result: Summary result dictionary
            base_filename: Base filename for all exports
        
        Returns:
            dict: Dictionary with paths to all exported files
        """
        try:
            exported_files = {}
            
            # Prepare metadata
            metadata = {
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Audio File': transcript_result.get('audio_metadata', {}).get('file_name', 'Unknown'),
                'Duration': transcript_result.get('audio_metadata', {}).get('duration_formatted', 'Unknown'),
                'Transcription Model': transcript_result.get('model_name', 'Unknown'),
                'Summary Model': summary_result.get('model_name', 'Unknown'),
                'Word Count (Transcript)': transcript_result.get('word_count', 0),
                'Word Count (Summary)': summary_result.get('word_count', 0)
            }
            
            # Export transcript as TXT
            transcript_txt = self.export_to_txt(
                transcript_result.get('formatted_transcript', ''),
                f"{base_filename}_transcript",
                metadata
            )
            exported_files['transcript_txt'] = transcript_txt
            
            # Export summary as TXT
            summary_txt = self.export_to_txt(
                summary_result.get('summary', ''),
                f"{base_filename}_summary",
                metadata
            )
            exported_files['summary_txt'] = summary_txt
            
            # Export combined Markdown
            combined_md_content = f"""## Transcript

{transcript_result.get('formatted_transcript', '')}

---

## Summary

{summary_result.get('summary', '')}
"""
            combined_md = self.export_to_markdown(
                combined_md_content,
                f"{base_filename}_complete",
                title="Class Lecture Transcription and Summary",
                metadata=metadata
            )
            exported_files['combined_markdown'] = combined_md
            
            # Export JSON with all data
            complete_data = {
                'metadata': metadata,
                'transcript': {
                    'text': transcript_result.get('formatted_transcript', ''),
                    'raw': transcript_result.get('transcript', ''),
                    'word_count': transcript_result.get('word_count', 0),
                    'model': transcript_result.get('model', ''),
                },
                'summary': {
                    'text': summary_result.get('summary', ''),
                    'type': summary_result.get('summary_type', ''),
                    'word_count': summary_result.get('word_count', 0),
                    'model': summary_result.get('model', ''),
                    'structured_data': summary_result.get('structured_data', {})
                },
                'audio_info': transcript_result.get('audio_metadata', {})
            }
            json_file = self.export_to_json(complete_data, f"{base_filename}_data")
            exported_files['json'] = json_file
            
            logger.info(f"Exported complete session: {len(exported_files)} files")
            return exported_files
        
        except Exception as e:
            logger.error(f"Error exporting complete session: {str(e)}")
            raise
    
    def _format_metadata_header(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata as a text header.
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            str: Formatted metadata header
        """
        lines = ["METADATA", "="*80]
        
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def generate_filename(
        self,
        base_name: Optional[str] = None,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate a filename for export.
        
        Args:
            base_name: Optional base name for the file
            include_timestamp: Whether to include timestamp
        
        Returns:
            str: Generated filename (without extension)
        """
        parts = []
        
        if base_name:
            # Clean base name
            base_name = base_name.replace(' ', '_')
            base_name = ''.join(c for c in base_name if c.isalnum() or c in ('_', '-'))
            parts.append(base_name)
        else:
            parts.append('lecture')
        
        if include_timestamp or EXPORT_SETTINGS.get('include_timestamp', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            parts.append(timestamp)
        
        return '_'.join(parts)
    
    def get_output_directory(self) -> str:
        """Get the output directory path."""
        return str(self.output_dir)


# Optional: PDF and DOCX export (requires additional libraries)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    
    class PDFExporter(FileExporter):
        """Extended exporter with PDF support"""
        
        def export_to_pdf(
            self,
            content: str,
            filename: str,
            title: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> str:
            """
            Export content to PDF file.
            
            Args:
                content: Text content to export
                filename: Output filename (without extension)
                title: Optional title for the document
                metadata: Optional metadata to include
            
            Returns:
                str: Path to exported file
            """
            try:
                output_path = self.output_dir / f"{filename}.pdf"
                
                # Create PDF
                doc = SimpleDocTemplate(
                    str(output_path),
                    pagesize=letter,
                    rightMargin=inch,
                    leftMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch
                )
                
                # Container for PDF elements
                story = []
                styles = getSampleStyleSheet()
                
                # Title style
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    alignment=TA_CENTER,
                    spaceAfter=30
                )
                
                # Add title
                if title:
                    story.append(Paragraph(title, title_style))
                    story.append(Spacer(1, 0.2*inch))
                
                # Add metadata
                if metadata:
                    meta_style = ParagraphStyle(
                        'Metadata',
                        parent=styles['Normal'],
                        fontSize=10,
                        textColor='grey'
                    )
                    for key, value in metadata.items():
                        story.append(Paragraph(f"<b>{key}:</b> {value}", meta_style))
                    story.append(Spacer(1, 0.3*inch))
                
                # Add content
                content_style = styles['Normal']
                content_style.alignment = TA_LEFT
                
                # Split content into paragraphs
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Handle special formatting
                        if para.startswith('##'):
                            # Header
                            para_text = para.replace('##', '').strip()
                            story.append(Paragraph(para_text, styles['Heading2']))
                        elif para.startswith('**') and para.endswith('**'):
                            # Bold section
                            para_text = para.replace('**', '').strip()
                            story.append(Paragraph(f"<b>{para_text}</b>", styles['Heading3']))
                        else:
                            # Regular paragraph
                            story.append(Paragraph(para, content_style))
                        story.append(Spacer(1, 0.1*inch))
                
                # Build PDF
                doc.build(story)
                
                logger.info(f"Exported to PDF: {output_path}")
                return str(output_path)
            
            except Exception as e:
                logger.error(f"Error exporting to PDF: {str(e)}")
                raise
    
    # Use PDFExporter if reportlab is available
    FileExporter = PDFExporter
    
except ImportError:
    logger.info("reportlab not installed. PDF export will not be available.")
    logger.info("Install with: pip install reportlab")


if __name__ == "__main__":
    # Test the exporter
    exporter = FileExporter()
    print(f"File Exporter initialized")
    print(f"Output directory: {exporter.get_output_directory()}")
    
    # Test filename generation
    filename = exporter.generate_filename("test_lecture")
    print(f"Generated filename: {filename}")
