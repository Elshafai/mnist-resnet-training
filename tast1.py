import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# تعريف التحويلات المطلوبة للصور
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # تحويل الصور من 1 قناة إلى 3 قنوات (RGB)
    transforms.Resize((224, 224)),  # تغيير الحجم ليُناسب نموذج ResNet
    transforms.ToTensor(),  # تحويل الصور إلى Tensor
    transforms.Normalize([0.5], [0.5])  # تطبيع البيانات
])

# تحميل بيانات MNIST
try:
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل البيانات: {e}")
    exit()

# تحميل البيانات باستخدام DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# عدد الفئات (0-9)
num_classes = 10

# التأكد من أن البيانات تم تحميلها بنجاح
print(f"عدد صور التدريب: {len(train_dataset)}")
print(f"عدد صور الاختبار: {len(test_dataset)}")


# عرض بعض الصور
def imshow(img):
    img = img / 2 + 0.5  # إزالة التطبيع
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# عرض بعض الصور مع تسميتها
try:
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print("Labels:", labels.numpy())
except Exception as e:
    print(f"❌ خطأ أثناء عرض الصور: {e}")

# تحميل نموذج ResNet18 مُدرب مسبقًا
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # تعديل الطبقة النهائية
except Exception as e:
    print(f"❌ خطأ أثناء تحميل النموذج: {e}")
    exit()

# نقل النموذج إلى وحدة المعالجة (GPU إذا كانت متاحة)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# تحديد دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# عدد العصور (Epochs)
num_epochs = 5

# تدريب النموذج
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# تقييم النموذج
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ دقة النموذج على بيانات الاختبار: {accuracy:.2f}%")