from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import logging
from config import Config
from modules.signal_manager import SignalManager
import asyncio
import threading

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.signal_manager = SignalManager()
        self.bot = Bot(token=self.token)
        self.application = ApplicationBuilder().token(self.token).build()
        self.application.add_handler(CommandHandler('signals', self.signals_command))
        self.application.add_handler(CommandHandler('performance', self.performance_command))
        
        # Event loop için thread-safe yaklaşım
        self._loop = None
        self._lock = threading.Lock()

    def _get_or_create_loop(self):
        """Thread-safe event loop yönetimi"""
        with self._lock:
            try:
                # Mevcut event loop'u kontrol et
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop kapalıysa yeni oluştur
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop
            except RuntimeError:
                # Event loop yoksa yeni oluştur
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop

    def send_signal(self, signal):
        """Thread-safe sinyal gönderimi"""
        try:
            msg = self.format_signal(signal)
            
            # Event loop'u güvenli şekilde al
            loop = self._get_or_create_loop()
            
            # Asenkron gönderim fonksiyonu
            async def send_message_async():
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id, 
                        text=msg, 
                        parse_mode='HTML'
                    )
                    self.logger.info('Telegram sinyal bildirimi gönderildi.')
                except Exception as e:
                    self.logger.error(f'Telegram mesaj gönderme hatası: {e}')
            
            # Event loop'ta çalıştır
            if loop.is_running():
                # Loop zaten çalışıyorsa, future olarak ekle
                asyncio.run_coroutine_threadsafe(send_message_async(), loop)
            else:
                # Loop çalışmıyorsa, çalıştır
                loop.run_until_complete(send_message_async())
                
        except Exception as e:
            self.logger.error(f'Telegram sinyal bildirimi gönderilemedi: {e}')

    def send_signal_notification(self, signal):
        """Sinyal bildirimi gönder (send_signal için alias)"""
        return self.send_signal(signal)

    def format_signal(self, signal):
        msg = f"<b>Yeni Sinyal</b>\n"
        msg += f"Coin: <b>{signal['symbol']}</b>\n"
        msg += f"Zaman Dilimi: <b>{signal['timeframe']}</b>\n"
        msg += f"Yön: <b>{signal['direction']}</b>\n"
        msg += f"AI Skoru: <b>{signal['ai_score']:.2f}</b>\n"
        msg += f"Tahmini Yükseliş: <b>{signal.get('predicted_gain', '-')}%</b>\n"
        msg += f"Tahmini Süre: <b>{signal.get('predicted_duration', '-')}</b>\n"
        msg += f"Tarih: <b>{signal['timestamp']}</b>\n"
        return msg

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        df = self.signal_manager.get_latest_signals(limit=10)
        if df.empty:
            await update.message.reply_text('Aktif sinyal yok.')
            return
        msg = '<b>Aktif Sinyaller</b>\n'
        for _, row in df.iterrows():
            msg += f"{row['symbol']} | {row['direction']} | Skor: {row['ai_score']:.2f} | {row['timestamp']}\n"
        await update.message.reply_text(msg, parse_mode='HTML')

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        df = self.signal_manager.load_signals()
        if df.empty:
            await update.message.reply_text('Geçmiş sinyal verisi yok.')
            return
        total = len(df)
        success = len(df[df['result'] == 'SUCCESS']) if 'result' in df.columns else 0
        avg_gain = df['realized_gain'].mean() if 'realized_gain' in df.columns else 0
        msg = f"Toplam Sinyal: <b>{total}</b>\nBaşarı Oranı: <b>{(success/total*100):.1f}%</b>\nOrtalama Yükseliş: <b>{avg_gain:.2f}%</b>"
        await update.message.reply_text(msg, parse_mode='HTML')

    def run(self):
        self.logger.info('Telegram botu başlatıldı.')
        self.application.run_polling() 